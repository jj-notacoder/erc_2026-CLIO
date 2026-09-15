"""Explicit lift-only force/velocity policies, real callbacks and fake time.

The Run25 nearest outlier (not its unlogged private gate snapshot) supplies the
8.695393 µm/s / 17.484 nm example. Synthetic 4 N histories below are explicitly
not a claim that Run25 had a fresh 3.5 N/50 ms window after rejection.
"""
from dataclasses import replace
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
from erc_phase1_solution import lift_pressure_gate as module
from erc_phase1_solution.adaptive_grasp import ForceSample
from test_fine_lift_pressure_gate import fixture
from test_contact_force_timing import force_message, LEFT, RIGHT, BOOK, MODEL


def floor_selected(monkeypatch):
    gate,n,now,wall,sent,events,make_joint,send=selected(monkeypatch,maximum=2e-6)
    n.lift_first_minimum_contact_force=2.5
    gate=module.LiftPressureGate(n,gate.checked,gate.reference)
    n._clear_target_contact_samples()
    for stamp in range(now.nanoseconds-60_000_000,now.nanoseconds+1,2_000_000):
        n._on_contacts(force_message(stamp,(LEFT,BOOK,2.591667186064023),(RIGHT,BOOK,2.824792890395198)))
    return gate,n,now,wall,sent,events,make_joint,send


@pytest.mark.parametrize('floor,allowed',[(2.5,True),(3.5,True),(2.499999,False),(0.,False),
    (float('nan'),False),(float('inf'),False),(30.,False)])
def test_real_node_floor_validation_block_and_unchanged_default(floor,allowed):
    from erc_phase1_solution import manipulation_node
    tree=ast.parse(Path(manipulation_node.__file__).read_text())
    constructor=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='__init__'
                     and any(isinstance(t,ast.Attribute) and t.attr=='lift_first_minimum_contact_force'
                             for t in ast.walk(n)))
    guard=next(n for n in constructor.body if isinstance(n,ast.If)
               and 'math.isfinite(self.lift_first_minimum_contact_force)' in ast.unparse(n.test))
    node=SimpleNamespace(lift_first_minimum_contact_force=floor,
                         lift_first_extraction_enabled=True,adaptive_contact_force_maximum=30.)
    def execute():exec(compile(ast.Module(body=[guard],type_ignores=[]),'real_node_floor_guard','exec'),dict(self=node,math=math))
    if allowed:execute()
    else:
        with pytest.raises(ValueError,match='>=2.5'):execute()
    assert "'lift_first_minimum_contact_force': 3.5" in Path(manipulation_node.__file__).read_text()


def test_selected_floor_two_point_five_admits_with_strict_velocity_and_default_three_point_five_does_not(monkeypatch):
    gate,n,now,_,sent,events,make_joint,send=floor_selected(monkeypatch)
    # Values match the recorded postprobe sample; history/context are synthetic.
    deliver(n,make_joint,-1.0290218099276091e-9)
    snapshot=gate._snapshot_locked()
    assert gate.maximum_gripper_velocity==2e-6 and gate.minimum_force==2.5
    assert gate._evaluate(snapshot,now.nanoseconds).verified
    n.lift_first_minimum_contact_force=3.5
    default=module.LiftPressureGate(n,gate.checked,gate.reference)
    assert not default._evaluate(snapshot,now.nanoseconds).verified
    assert gate.send(send)=='accepted-future'
    assert len(sent)==1 and events[-1][1]['minimum_force']==2.5
    assert events[-1][1]['maximum_gripper_velocity_mps']==2e-6


@pytest.mark.parametrize('kind',['latest','future','backfilled'])
def test_selected_floor_rejects_new_subfloor_force_at_final_admission(monkeypatch,kind):
    gate,n,now,wall,sent,_,_,send=floor_selected(monkeypatch)
    def update():
        stamp=now.nanoseconds+2_000_000 if kind=='future' else now.nanoseconds-25_000_000 if kind=='backfilled' else now.nanoseconds
        n._on_contacts(force_message(stamp,(LEFT,BOOK,2.49)))
    wall.hook=update
    with pytest.raises(module.LiftPressureRejected):gate._attempt(send,wall.seconds+5.)
    assert sent==[]


def test_selected_floor_keeps_original_two_micrometre_velocity_rejection(monkeypatch):
    gate,n,_,_,sent,events,make_joint,send=floor_selected(monkeypatch)
    deliver(n,make_joint,8.695393e-6)
    with pytest.raises(module.LiftPressureRejected,match='gripper_not_stationary'):gate.send(send)
    assert not sent and events[-1][1]['admission_snapshot']['maximum_gripper_velocity_mps']==2e-6


def selected(monkeypatch, maximum=1e-5):
    gate,n,now,wall,sent,events,make_joint,send=fixture(monkeypatch)
    n.lift_first_maximum_gripper_velocity_mps=maximum
    n.adaptive_contact_force_maximum=30.
    gate=module.LiftPressureGate(n,gate.checked,gate.reference)
    return gate,n,now,wall,sent,events,make_joint,send


def deliver(n,make_joint,velocity,*,position=None):
    message=make_joint()
    index=message.name.index('gripper_left_finger_joint')
    message.velocity[index]=velocity
    if position is not None:message.position[index]=position
    n._on_joint_state(message)


def test_default_remains_two_and_both_checks_use_selected_ten(monkeypatch):
    old,n,now,_,_,_,make_joint,_=fixture(monkeypatch)
    deliver(n,make_joint,8.695393e-6,position=.0183+17.484e-9)
    snapshot=old._snapshot_locked()
    assert old.maximum_gripper_velocity==2e-6
    with pytest.raises(module.LiftPressureRejected,match='gripper_not_stationary'):
        old._hard_check(snapshot,now.nanoseconds,allow_future=False)
    assert not old._evaluate(snapshot,now.nanoseconds).verified
    n.lift_first_maximum_gripper_velocity_mps=1e-5
    new=module.LiftPressureGate(n,old.checked,old.reference)
    new._hard_check(snapshot,now.nanoseconds,allow_future=False)
    assert new._evaluate(snapshot,now.nanoseconds).verified


@pytest.mark.parametrize('value',[0.,-1e-6,1.0000001e-5,float('nan'),float('inf'),-float('inf')])
def test_invalid_maximum_rejected_before_send(monkeypatch,value):
    old,n,_,_,sent,_,_,_=fixture(monkeypatch)
    n.lift_first_maximum_gripper_velocity_mps=value
    with pytest.raises(ValueError):module.LiftPressureGate(n,old.checked,old.reference)
    assert sent==[]


def test_selected_limit_frozen_per_permit_and_independent_of_micro(monkeypatch):
    gate,n,_,_,_,_,_,_=selected(monkeypatch)
    n.lift_first_maximum_gripper_velocity_mps=2e-6
    n.fine_gripper_limits=type('Limits',(),{'micro_stationary_velocity':1.})()
    assert gate.maximum_gripper_velocity==1e-5


def test_real_callback_nearest_recorded_velocity_admits_once_with_synthetic_force(monkeypatch):
    gate,n,_,_,sent,events,make_joint,send=selected(monkeypatch)
    deliver(n,make_joint,8.695393e-6,position=.0183+17.484e-9)
    assert gate.send(send)=='accepted-future'
    assert len(sent)==1
    assert events[-1][1]['maximum_gripper_velocity_mps']==1e-5
    with pytest.raises(module.LiftPressureRejected,match='already_consumed'):gate.send(send)
    assert len(sent)==1


def test_late_over_limit_callback_blocks_and_logs_exact_rejected_input(monkeypatch):
    gate,n,now,wall,sent,events,make_joint,send=selected(monkeypatch)
    target=.0183+17.484e-9
    def late():deliver(n,make_joint,10.01e-6,position=target)
    wall.hook=late
    with pytest.raises(module.LiftPressureRejected,match='gripper_not_stationary'):gate.send(send)
    assert not sent
    rejected=events[-1][1]
    snapshot=rejected['admission_snapshot']
    assert snapshot['phase'] in ('catchup_live','dispatch_live')
    assert snapshot['evaluation_ros_ns']==now.nanoseconds
    assert snapshot['feedback']['position_m']==target
    assert snapshot['feedback']['velocity_mps']==10.01e-6
    assert snapshot['feedback']['effort']==-.1
    assert snapshot['maximum_gripper_velocity_mps']==1e-5
    assert snapshot['elapsed_wall_seconds']<5.
    assert snapshot['force_sides']['left']['latest_force_n']==4.
    assert snapshot['force_sides']['right']['last_producer_stamp_ns']>0
    assert len(snapshot['recent_feedback'])<=8
    assert all(len(side['recent_samples'])<=8 for side in snapshot['force_sides'].values())
    # Later callbacks cannot replace the private input already emitted.
    deliver(n,make_joint,0.,position=.0183)
    assert snapshot['feedback']['velocity_mps']==10.01e-6


@pytest.mark.parametrize('fault,reason',[
    ('force','force_overload'),('effort','effort_overload'),('delta','effort_delta_overload'),
    ('identity','target_identity_invalid'),('stale','named_contact_stale'),
    ('width','width_invalid'),('arm','arm_not_stationary'),('context','collision_context_changed'),
    ('base','registered_base_moved'),('finger','finger_geometry_changed'),('cancel','cancelled')])
def test_selected_velocity_never_bypasses_other_guards(monkeypatch,fault,reason):
    gate,n,now,_,sent,events,make_joint,send=selected(monkeypatch)
    deliver(n,make_joint,8.695393e-6)
    if fault=='force':n._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,31.)))
    elif fault in ('effort','delta'):
        n._gripper_feedback_samples.append(replace(n._gripper_feedback_samples[-1],effort=5. if fault=='effort' else 3.2))
    elif fault=='identity':n._target_book_model='book_col_2_row_1_red'
    elif fault=='stale':n._book_contact_force_samples[MODEL]=([ForceSample(now.nanoseconds-151_000_000,4.)],)*2
    elif fault=='width':n._gripper_feedback_samples.append(replace(n._gripper_feedback_samples[-1],position=.014))
    elif fault=='arm':n._joint_velocities['arm_left_1_joint']=.0011
    elif fault=='context':n.joints['head_1_joint']=.002
    elif fault=='base':n._staging_odom['pose']=(2.003,-.1,.02)
    elif fault=='finger':n.joints['gripper_left_finger_joint']+=.00002
    elif fault=='cancel':n._cancel.set()
    with pytest.raises(module.LiftPressureRejected,match=reason):gate.send(send)
    assert sent==[]
    assert events[-1][0]=='lift_first_pressure_rejected'
    assert events[-1][1]['admission_snapshot']['feedback']['velocity_mps']==8.695393e-6


def test_weak_pressure_still_needs_fresh_fifty_ms_and_original_five_wall_deadline(monkeypatch):
    gate,n,now,wall,sent,events,make_joint,send=selected(monkeypatch)
    def weak():
        deliver(n,make_joint,8e-6)
        n._staging_odom['stamp_ns']=now.nanoseconds
        n._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,.3),(RIGHT,BOOK,.4)))
    weak();wall.hook=weak
    began=wall.seconds
    with pytest.raises(module.LiftPressureRejected,match='wait_timeout|pressure_admission_deadline'):gate.send(send)
    assert sent==[] and wall.seconds-began<=5.0021
    snapshot=events[-1][1]['admission_snapshot']
    assert snapshot['overall_wall_timeout_seconds']==5.
    assert snapshot['elapsed_wall_seconds']>=5.
    assert snapshot['minimum_force_span_seconds']==.05
    assert sum(event=='lift_first_pressure_waiting' for event,_ in events)==1


def test_selected_velocity_does_not_accept_short_force_pulses(monkeypatch):
    gate,n,now,wall,sent,_,make_joint,send=selected(monkeypatch)
    start=now.nanoseconds
    def pulsed():
        deliver(n,make_joint,8e-6)
        n._staging_odom['stamp_ns']=now.nanoseconds
        high=((now.nanoseconds-start)//1_000_000)%40<34
        n._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,4. if high else .16),(RIGHT,BOOK,5. if high else .0068)))
    # Remove old synthetic strong proof before starting the repeated34ms pulses.
    n._clear_target_contact_samples()
    pulsed();wall.hook=pulsed
    with pytest.raises(module.LiftPressureRejected):gate.send(send)
    assert not sent


def test_snapshot_export_problem_cannot_hide_original_negative_reason(monkeypatch):
    gate,n,_,_,sent,events,make_joint,send=selected(monkeypatch)
    deliver(n,make_joint,11e-6)
    monkeypatch.setattr(gate,'_format_rejection_snapshot',lambda:(_ for _ in ()).throw(ValueError('broken diagnostic')))
    with pytest.raises(module.LiftPressureRejected,match='gripper_not_stationary'):gate.send(send)
    assert sent==[] and events[-1][1]['admission_snapshot']['snapshot_export_error']=='broken diagnostic'
