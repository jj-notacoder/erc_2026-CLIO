"""Stock q0 settling uses configured endpoint tolerance, not fixed-hold 10um."""
import copy
from dataclasses import replace
import numpy as np
import pytest
pytest.importorskip('rclpy')
from erc_phase1_solution import lift_first_extraction as lift
from erc_phase1_solution import lift_pressure_gate as gates
from test_fine_lift_pressure_gate import fixture
from test_lift_first_measured_collision_context import _recheck_node, _feed, _check
from test_lift_first_integration import _enabled
from test_shutdown import _mocked_pick_trace

MASTER='gripper_left_finger_joint'
# Parent Run27 status/feedback values: actuator coordinate, not physical padgap.
RECORDED_REFERENCE=17.957285e-6
RECORDED_CURRENT=2.887e-9

@pytest.mark.parametrize('stock', [False,True])
@pytest.mark.parametrize('fraction', [.999,1.001])
def test_measured_geometry_master_boundary_preserves_default_and_no_commands(monkeypatch,stock,fraction):
    n=_recheck_node();n.stock_gripper_close_diagnostic_enabled=stock
    n.adaptive_endpoint_tolerance=.0006
    limit=.0006 if stock else .00001
    _feed(n,{MASTER:0.})
    original=n._lift_first_measurements()
    def validate(*args,**kwargs):
        assert kwargs['aperture']==0.
        _feed(n,{MASTER:limit*fraction},1_004_000_000)
        return {}
    monkeypatch.setattr(lift,'validate_lift_first_route',validate)
    if fraction>1:
        with pytest.raises(RuntimeError,match='finger moved') as error:_check(n,original)
        assert error.value.master_geometry['master_geometry_tolerance_m']==limit
        assert error.value.master_geometry['current_master_position_m']==limit*fraction
    else:
        result=_check(n,original)
        assert result['geometry_reference_master_position_m']==0.
        fields=n.statuses[-1][1]
        assert fields['master_geometry_tolerance_m']==limit
        assert fields['current_master_position_m']==limit*fraction

@pytest.mark.parametrize('stock',[False,True])
def test_recorded_run27_actuator_settling_rejects_fixed_hold_but_accepts_stock(monkeypatch,stock):
    n=_recheck_node();n.stock_gripper_close_diagnostic_enabled=stock
    n.adaptive_endpoint_tolerance=.0006
    _feed(n,{MASTER:RECORDED_REFERENCE});original=n._lift_first_measurements()
    calls=[]
    def validate(*args,**kwargs):
        calls.append(kwargs['aperture'])
        _feed(n,{MASTER:RECORDED_CURRENT},1_004_000_000)
        return {}
    monkeypatch.setattr(lift,'validate_lift_first_route',validate)
    if not stock:
        with pytest.raises(RuntimeError,match='finger moved'):_check(n,original)
    else:
        result=_check(n,original)
        assert result['fingers'][MASTER]==RECORDED_CURRENT
        assert result['geometry_reference_master_position_m']==RECORDED_REFERENCE
        fields=n.statuses[-1][1]
        assert fields['geometry_reference_master_position_m']==RECORDED_REFERENCE
        assert 'no geometry recomputation' in fields['master_geometry_scope']
    assert calls==[RECORDED_REFERENCE]

@pytest.mark.parametrize('stock,delta,accepted',[(False,17.95e-6,False),(True,17.95e-6,True),
    (True,.000599,True),(True,.000601,False)])
def test_actual_pick_after_contact_probe_uses_same_selected_master_tolerance(monkeypatch,stock,delta,accepted):
    def configure(n,seen):
        n.stock_gripper_close_diagnostic_enabled=stock;n.adaptive_endpoint_tolerance=.0006
        original=n._lift_first_measurements
        seen['probe_finished']=False
        def probe(command,phase,**kwargs):
            if phase=='before_initial_shelf_lift':seen['probe_finished']=True
            return True
        n._fresh_retention_probe=probe
        def measurements(reference=None):
            sample=copy.deepcopy(original(reference))
            if seen['probe_finished']:sample['fingers'][MASTER]+=delta
            return sample
        n._lift_first_measurements=measurements
        seen['moves']=[];move=n._move_arm_solution
        n._move_arm_solution=lambda q,t:seen['moves'].append(q[1]) or move(q,t)
        seen['events']=[];publish=n._publish_status
        n._publish_status=lambda event,**fields:seen['events'].append((event,fields)) or publish(event,**fields)
    seen,setup=_enabled(monkeypatch,configure=configure)
    if accepted:
        result=_mocked_pick_trace([.70,0.,1.58],configure_node=setup)
        assert result['succeeded'] and 40 in seen['moves']
    else:
        with pytest.raises(RuntimeError,match='pick_recovery_failed'):
            _mocked_pick_trace([.70,0.,1.58],configure_node=setup)
        assert 40 not in seen['moves']
        rejection=next(f for e,f in seen['events'] if e=='lift_first_preflight_rejected')
        assert rejection['master_geometry_tolerance_m']==(.0006 if stock else .00001)

@pytest.mark.parametrize('stock,delta,accepted',[(False,17.95e-6,False),(True,17.95e-6,True),
    (True,.000599,True),(True,.000601,False)])
def test_final_locked_gate_tolerance_and_exact_diagnostic_snapshot(monkeypatch,stock,delta,accepted):
    gate,n,now,wall,sent,events,joint,send=fixture(monkeypatch)
    n.stock_gripper_close_diagnostic_enabled=stock;n.adaptive_endpoint_tolerance=.0006
    n.adaptive_velocity_tolerance=.003
    gate=gates.LiftPressureGate(n,gate.checked,gate.reference)
    prior=n.joints[MASTER]
    # Drift arrives during fixed-clock catchup, exercising the live admission.
    def drift():
        n.joints[MASTER]=prior+delta;n._on_joint_state(joint())
    wall.hook=drift
    if accepted:
        assert gate.send(send)=='accepted-future'
        assert len(sent)==1
    else:
        with pytest.raises(gates.LiftPressureRejected,match='finger_geometry_changed'):gate.send(send)
        assert not sent
    fields=events[-1][1]
    assert fields['master_comparison_reference_position_m']==prior
    assert fields['current_master_position_m']==prior+delta
    assert fields['master_geometry_tolerance_m']==(.0006 if stock else .00001)
    if not accepted:
        saved=fields['admission_snapshot']['master_geometry']
        n.joints[MASTER]=99.
        assert saved['current_master_position_m']==prior+delta

def test_stock_gate_keeps_passive_freshness_and_one_milliradian_position_bound(monkeypatch):
    gate,n,now,wall,sent,events,joint,send=fixture(monkeypatch)
    n.stock_gripper_close_diagnostic_enabled=True;n.adaptive_endpoint_tolerance=.0006
    n.adaptive_velocity_tolerance=.003
    passive=gate.checked['modeled_finger_joints'][0]
    n.joints[passive]=.1;n._on_joint_state(joint())
    gate.checked['fingers'][passive]=.1
    gate.checked['reported_passive_joints']=[passive];gate.checked['modeled_finger_joints']=[]
    gate=gates.LiftPressureGate(n,gate.checked,gate.reference)
    n.joints[passive]=.101001;n._on_joint_state(joint())
    with pytest.raises(gates.LiftPressureRejected,match='finger_geometry_changed:'+passive):gate.send(send)
    assert not sent

def test_stock_gate_freezes_the_configured_value_not_a_new_hardcoded_limit(monkeypatch):
    gate,n,*_=fixture(monkeypatch)
    n.stock_gripper_close_diagnostic_enabled=True;n.adaptive_velocity_tolerance=.003
    n.adaptive_endpoint_tolerance=.0003
    gate=gates.LiftPressureGate(n,gate.checked,gate.reference)
    n.adaptive_endpoint_tolerance=.0006
    assert gate.master_geometry_tolerance==.0003
