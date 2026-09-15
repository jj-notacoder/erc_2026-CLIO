"""One absolute1500wall budget, with real controller/evaluator and fake ROS time.

No Node is constructed or spun; no actuator is started. Selected diagnostic
budgets do not represent competition timing or physical pickup success.
"""
from dataclasses import asdict

import pytest
pytest.importorskip('rclpy')
from erc_phase1_solution import fine_gripper_close as fine
from test_fine_gripper_close import micro_fixture, pressure_fixture, set_feedback
from test_fine_pressure_resettle import records
from test_contact_force_timing import LEFT, RIGHT, force_message


def selected_limits():
    return fine.FineGripLimits(total_wall_seconds=1500.,step_wall_seconds=15.,
        maximum_force=30.,minimum_force=8.,micro_motion_seconds=.4,
        micro_preload_steps=300,micro_preload_step=.25e-6,micro_preload_limit=75e-6,
        micro_position_error=50e-9,micro_stationary_velocity=10e-6)


def active(monkeypatch, *, slow=False):
    values=pressure_fixture(monkeypatch,limits=selected_limits())
    c,node,now,clock,_,_=values
    c._fine_resettle_enabled=True
    c._fine_motion_end_ns=now.nanoseconds
    c._fine_step_command=c._fine_command
    c._fine_step_motion_end_ns=c._fine_motion_end_ns
    c._fine_step_last_ros_ns=now.nanoseconds
    book=c.expected_model+'::book_base_link::collision'
    def sensors():
        set_feedback(node,now,c._fine_command)
        node._on_contacts(force_message(now.nanoseconds,(LEFT,book,1.),(RIGHT,book,1.3)))
    if slow:
        def sleep(seconds):
            clock.wall+=seconds
            if clock.advancing:now.nanoseconds+=round(seconds*1e9*.025)
            if clock.on_sleep:clock.on_sleep()
        clock.sleep=sleep
    clock.on_sleep=sensors
    sensors()
    return values


def test_1500_is_optional_and_changes_only_total_wall_budget():
    default=fine.FineGripLimits()
    extended=fine.FineGripLimits(total_wall_seconds=1500.)
    assert default.total_wall_seconds==300. and default.step_wall_seconds==5.
    assert {k for k,v in asdict(default).items() if asdict(extended)[k]!=v}=={'total_wall_seconds'}
    selected=selected_limits()
    assert selected.step_wall_seconds==15. and selected.preclose_wall_seconds==10.
    assert selected.micro_motion_seconds==.4 and selected.dwell_seconds==.2
    assert selected.micro_preload_limit==75e-6 and selected.maximum_force==30.


@pytest.mark.parametrize('seconds',[0.,-1.,1500.00001,float('nan'),float('inf'),-float('inf')])
def test_total_wall_limit_remains_finite_positive_and_bounded(seconds):
    with pytest.raises(ValueError):fine.FineGripLimits(total_wall_seconds=seconds)


def test_total_deadline_created_once_survives_multiple_steps_and_rechecks(monkeypatch):
    c,node,now,clock,messages,statuses=micro_fixture(monkeypatch,limits=selected_limits())
    assert c._fine_wall_deadline==1500.
    clock.wall=1497.
    book=c.expected_model+'::book_base_link::collision'
    def sensors():
        set_feedback(node,now,c._fine_command)
        node._on_contacts(force_message(now.nanoseconds,(LEFT,book,1.),(RIGHT,book,1.3)))
    clock.on_sleep=sensors
    sensors()
    observed=[]
    original=c._wait_motion_and_stationary
    def wait(*args,**kwargs):
        observed.append((c._fine_wall_deadline,kwargs.get('wall_deadline')))
        return original(*args,**kwargs)
    c._wait_motion_and_stationary=wait
    c._run_micro_preload(finalize=True)
    assert len(records(statuses,'public_position_command'))>=2
    assert len(records(statuses,'pressure_recheck_started'))>=2
    assert observed and all(total==1500. and deadline<=1500. for total,deadline in observed)
    assert c._fine_wall_deadline==1500. and clock.wall<=1500.0021
    assert c._fine_fault in ('fine_grip_wall_timeout','step_wall_timeout')
    assert c._fine_result is None
    before=len(messages)
    with c._fine_lock:assert not c._publish_once_locked(c._fine_command-.25e-6,.4,'close')
    assert len(messages)==before


def test_fresh_same_target_recheck_cannot_extend_expiring_total(monkeypatch):
    c,_,_,clock,messages,statuses=active(monkeypatch)
    clock.wall=1499.98
    target,epoch=c._fine_command,c._fine_micro_stationary_start_ns
    evidence=c._micro_pressure_snapshot(2,target,wall_deadline=1514.98)
    assert evidence is not None and not evidence[1].verified
    assert c._recheck_micro_pressure(2,target,evidence,wall_deadline=1514.98) is None
    assert c._fine_wall_deadline==1500. and clock.wall<=1500.0021
    assert c._fine_micro_stationary_start_ns==epoch and c._fine_command==target
    assert len(messages)==1 and not records(statuses,'public_position_command')
    assert c._fine_fault in ('fine_grip_wall_timeout','step_wall_timeout')


def test_resettle_requires_full_dwell_but_is_clipped_to_original_total(monkeypatch):
    c,node,now,clock,messages,statuses=active(monkeypatch,slow=True)
    clock.wall=1494.
    set_feedback(node,now,c._fine_command,11e-6)
    assert not c._resettle_micro_epoch(3,c._fine_command,'joint_still_moving',wall_deadline=1509.)
    assert c._fine_wall_deadline==1500. and clock.wall<=1500.0021
    assert not records(statuses,'stationary_endpoint') and len(messages)==1
    assert not records(statuses,'public_position_command')


def test_two_resettles_share_the_original_fifteen_second_step_before_total(monkeypatch):
    c,node,now,clock,messages,statuses=active(monkeypatch,slow=True)
    clock.wall=1484.
    set_feedback(node,now,c._fine_command,11e-6)
    assert c._resettle_micro_epoch(3,c._fine_command,'joint_still_moving',wall_deadline=1499.)
    assert 1492.<clock.wall<1499.
    set_feedback(node,now,c._fine_command,11e-6)
    assert not c._resettle_micro_epoch(3,c._fine_command,'joint_still_moving',wall_deadline=1499.)
    assert c._fine_fault=='step_wall_timeout' and clock.wall<=1499.0021
    assert c._fine_wall_deadline==1500.
    assert len(records(statuses,'stationary_endpoint'))==1
    assert len(records(statuses,'pressure_epoch_invalidated'))==2
    assert len(messages)==1 and not records(statuses,'public_position_command')


def test_fifteen_seconds_remains_the_per_step_maximum():
    with pytest.raises(ValueError):fine.FineGripLimits(total_wall_seconds=1500.,step_wall_seconds=15.00001)


@pytest.mark.parametrize('kind,reason',[('cancel','cancelled'),('clock','clock_stalled')])
def test_longer_total_does_not_relax_immediate_cancel_or_clock_stall(monkeypatch,kind,reason):
    c,node,now,clock,messages,statuses=active(monkeypatch,slow=True)
    clock.wall=1000.
    set_feedback(node,now,c._fine_command,11e-6)
    if kind=='cancel':node._cancel.set()
    else:clock.advancing=False
    assert not c._resettle_micro_epoch(3,c._fine_command,'joint_still_moving',wall_deadline=1015.)
    assert c._fine_fault==reason and clock.wall<=1001.004
    assert len(messages)==1 and not records(statuses,'public_position_command')
