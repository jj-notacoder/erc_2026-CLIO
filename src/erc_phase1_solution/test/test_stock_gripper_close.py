"""Stock public-command diagnostic: real callbacks/messages with fake time."""
from dataclasses import replace
import ast
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest
pytest.importorskip('rclpy')
from erc_phase1_solution import stock_gripper_close as stock
from erc_phase1_solution import lift_pressure_gate as gate_module
from erc_phase1_solution import manipulation_node as node_module
from erc_phase1_solution.adaptive_grasp import ForceSample,GripperFeedback
from test_fine_lift_pressure_gate import fixture
from test_contact_force_timing import force_message,LEFT,RIGHT,BOOK,MODEL


def selected(monkeypatch):
    gate,n,now,wall,sent,events,joint,send=fixture(monkeypatch)
    n.stock_gripper_close_diagnostic_enabled=True
    n.adaptive_velocity_tolerance=.003
    n.adaptive_endpoint_tolerance=.0006
    n._stock_close_controller=None
    n._adaptive_close_active=False
    n._adaptive_motion_started=False
    n._adaptive_motion_halt_reason=None
    n._adaptive_overload_latched=None
    n._adaptive_hold_sent=False
    n.gripper_transport_lock=.015
    gate=gate_module.LiftPressureGate(n,gate.checked,gate.reference)
    return gate,n,now,wall,sent,events,joint,send


@pytest.mark.parametrize('force,effort,master',[(40.,5.,.018),(1000.,10.,0.),(0.,.1,.018)])
def test_explicit_stock_gate_admits_actual_contact_without_amplitude_or_width_policy(monkeypatch,force,effort,master):
    gate,n,now,wall,sent,events,joint,send=selected(monkeypatch)
    n._clear_target_contact_samples()
    n.joints['gripper_left_finger_joint']=master
    message=joint();i=message.name.index('gripper_left_finger_joint')
    message.effort[i]=effort;message.velocity[i]=8.7e-6
    n._on_joint_state(message)
    gate.checked['fingers']['gripper_left_finger_joint']=master
    for stamp in range(now.nanoseconds-60_000_000,now.nanoseconds+1,2_000_000):
        n._on_contacts(force_message(stamp,(LEFT,BOOK,force),(RIGHT,BOOK,force)))
    assert getattr(n,'_held_grip_sensor_fault',None) is None
    assert gate.send(send)=='accepted-future'
    assert len(sent)==1 and events[-1][1]['contact_only'] is True
    assert events[-1][1]['left_force']==force
    assert gate.maximum_gripper_velocity==.003


@pytest.mark.parametrize('kind',['wrong','nan','missing_velocity','cancel','contact_lost','arm_moving','context'])
def test_stock_mode_retains_live_negative_admission_guards(monkeypatch,kind):
    gate,n,now,wall,sent,events,joint,send=selected(monkeypatch)
    def bad():
        if kind=='wrong':n._on_contacts(force_message(now.nanoseconds,(LEFT,'book_col_4_row_2_red::link::collision',100.)))
        elif kind=='nan':n._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,float('nan'))))
        elif kind=='missing_velocity':
            m=joint();m.velocity=[];n._on_joint_state(m)
        elif kind=='cancel':n._cancel.set()
        elif kind=='contact_lost':n._payload_hazard_latched='contact_lost'
        elif kind=='arm_moving':
            m=joint();m.velocity[m.name.index('arm_left_1_joint')]=.002;n._on_joint_state(m)
        elif kind=='context':n.joints['head_1_joint']=.002
    wall.hook=bad
    with pytest.raises(gate_module.LiftPressureRejected):gate.send(send)
    assert not sent


def close_fixture(monkeypatch):
    gate,n,now,wall,sent,events,joint,send=selected(monkeypatch)
    n._adaptive_close_active=True
    n._transport_lock_engaged=False
    n._target_book_model=None
    n._adaptive_force_peaks={}
    n.adaptive_contact_force_minimum=.05
    n._adaptive_effort_baseline_value=0.
    n._stock_close_controller=stock.StockGripperClose(n)
    n.gripper_pub=SimpleNamespace(publish=lambda message:sent.append(message))
    def sensors():
        message=joint()
        message.effort[message.name.index('gripper_left_finger_joint')]=getattr(n,'_test_effort',-.1)
        n._on_joint_state(message)
        n._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,80.)))
        n._on_contacts(force_message(now.nanoseconds,(RIGHT,BOOK,90.)))
    wall.hook=sensors
    return n._stock_close_controller,n,now,wall,sent,events,sensors


def test_real_public_trajectory_q_zero_one_second_never_replaced_by_success_hold(monkeypatch):
    controller,n,now,wall,sent,events,sensors=close_fixture(monkeypatch)
    result=controller.run()
    assert result[0] and len(sent)==1
    assert sent[0].joint_names==['gripper_left_finger_joint']
    assert list(sent[0].points[0].positions)==[0.]
    assert sent[0].points[0].time_from_start.sec==1
    assert sent[0].points[0].time_from_start.nanosec==0
    assert now.nanoseconds>=controller.command_end+200_000_000
    assert n._transport_lock_engaged and not n._adaptive_hold_sent
    assert events[-1][1]['control_mode']=='stock_public_position'
    assert events[-1][1]['pressure_qualified'] is False


@pytest.mark.parametrize('fault',['cancel','wrong','invalid_effort','stale','clock_reversed'])
def test_live_fault_interrupts_single_stock_command_and_no_duplicate_can_overwrite_hold(monkeypatch,fault):
    controller,n,now,wall,sent,events,sensors=close_fixture(monkeypatch)
    initial=now.nanoseconds
    def change():
        sensors()
        if now.nanoseconds-initial<20_000_000:return
        if fault=='cancel':n._cancel.set()
        elif fault=='wrong':n._on_contacts(force_message(now.nanoseconds,(LEFT,'shelf::link::collision',1.)))
        elif fault=='invalid_effort':
            n._gripper_feedback_samples.append(replace(n._gripper_feedback_samples[-1],effort=float('nan')))
        elif fault=='stale':n._gripper_feedback_samples.append(replace(n._gripper_feedback_samples[-1],stamp_ns=now.nanoseconds-200_000_000))
        elif fault=='clock_reversed':now.nanoseconds=initial-1
    wall.hook=change
    result=controller.run()
    assert not result[0]
    assert list(sent[0].points[0].positions)==[0.]
    assert len(sent)<=2
    if len(sent)==2:assert list(sent[1].points[0].positions)!=[0.]
    assert n._adaptive_hold_sent


def test_original_stock_wall_deadline_does_not_renew(monkeypatch):
    controller,n,now,wall,sent,events,sensors=close_fixture(monkeypatch)
    # Fresh clock advances, but far slower than wall time. It is not a stall.
    def slow(seconds):
        wall.seconds+=seconds*20
        now.nanoseconds+=round(seconds*1e9)
        sensors()
    monkeypatch.setattr(stock.time,'sleep',slow)
    assert not controller.run()[0]
    assert 15<=wall.seconds-controller.started_wall<15.1
    assert n._adaptive_motion_halt_reason=='stock_close_wall_timeout'


def test_high_effort_active_callback_is_logged_without_solution_amplitude_hold(monkeypatch):
    controller,n,now,wall,sent,events,sensors=close_fixture(monkeypatch)
    n._test_effort=8.
    assert controller.run()[0]
    assert len(sent)==1 and n._adaptive_overload_latched is None
    assert n._stock_effort_peak==8.


def test_transient_invalid_effort_callback_latches_before_later_valid_feedback(monkeypatch):
    controller,n,now,wall,sent,events,sensors=close_fixture(monkeypatch)
    n._test_effort=float('nan');sensors()
    n._test_effort=.1;sensors()
    assert n._adaptive_overload_latched=='invalid_stock_joint_feedback'
    with pytest.raises(RuntimeError,match='invalid_stock_joint_feedback'):controller.guard()


def test_wrong_contact_before_bilateral_is_latched_despite_later_good_frame(monkeypatch):
    controller,n,now,wall,sent,events,sensors=close_fixture(monkeypatch)
    n._on_contacts(force_message(now.nanoseconds,(LEFT,'shelf::link::collision',1.)))
    sensors()
    assert n._adaptive_overload_latched
    assert n._adaptive_motion_halt_reason is None
    with pytest.raises(RuntimeError):controller.guard()


@pytest.mark.parametrize('q',[0.,.014,.0183])
def test_pinch_retention_uses_master_coordinate_and_named_contacts_not_pad_width(monkeypatch,q):
    gate,n,now,wall,sent,events,joint,send=selected(monkeypatch)
    n.joints['gripper_left_finger_joint']=q
    assert n._pinch_sample(max_age=.15)[0]
    n._right_target_contact_ns=0
    assert not n._pinch_sample(max_age=.15)[0]


def test_constructor_policy_rejects_accidental_fine_or_interval_combination():
    tree=ast.parse(Path(node_module.__file__).read_text())
    constructor=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
        and n.name=='__init__' and 'self.stock_gripper_close_diagnostic_enabled' in ast.unparse(n))
    check=next(n for n in constructor.body if isinstance(n,ast.If)
        and ast.unparse(n.test).startswith('self.stock_gripper_close_diagnostic_enabled and'))
    for fine,interval,lift,okay in ((False,False,True,True),(True,False,True,False),
                                  (False,True,True,False),(False,False,False,False)):
        node=SimpleNamespace(stock_gripper_close_diagnostic_enabled=True,
            fine_gripper_close_enabled=fine,preclose_aperture_geometry_enabled=interval,
            lift_first_extraction_enabled=lift)
        def run():exec(compile(ast.Module(body=[check],type_ignores=[]),'real_stock_constructor_guard','exec'),{'self':node})
        if okay:run()
        else:
            with pytest.raises(ValueError):run()


def test_default_lift_policy_remains_strict(monkeypatch):
    gate,n,now,wall,sent,events,joint,send=fixture(monkeypatch)
    assert not gate.stock_mode and gate.minimum_force==3.5 and gate.maximum_gripper_velocity==2e-6
    n._gripper_feedback_samples.append(replace(n._gripper_feedback_samples[-1],effort=5.))
    with pytest.raises(gate_module.LiftPressureRejected,match='effort_overload'):gate.send(send)
    assert not sent
