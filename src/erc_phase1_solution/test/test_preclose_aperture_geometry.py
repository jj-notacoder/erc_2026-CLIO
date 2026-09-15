"""Outside overlay: attempt binding and actual pickup/admission ordering.

No real model is constructed here. Synthetic certificates exercise evidence
handling only; the separate real-model replay owns geometric verdict testing.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
from erc_phase1_solution import preclose_aperture_geometry as m
from erc_phase1_solution.aperture_interval_geometry import IntervalCertificate, _canonical, _policy_values
from erc_phase1_solution.motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS
from erc_phase1_solution.manipulation_node import ManipulationNode
from erc_phase1_solution import lift_pressure_gate
from test_lift_first_integration import _enabled
from test_shutdown import _mocked_pick_trace
from test_fine_lift_pressure_gate import fixture as pressure_fixture

_PRODUCTION_MODEL_RESOLVER = m._model


class Model:
    asset_binding = {'test_fixture': 'synthetic_not_official_proof'}
    def verify_assets_unchanged(self): pass
    def verify_node(self, node): pass


def _sample(q=.069):
    names=(*IK_JOINTS,*RIGHT_ARM_JOINTS,'head_1_joint','head_2_joint')
    joints=dict.fromkeys(names,0.);joints[m.MASTER]=q
    return dict(left=np.zeros(8),joints=joints,base_pose=np.zeros(3),stamp_ns=1_000_000_000,
        fingers={m.MASTER:q},reported_passive_joints=[],modeled_finger_joints=['gripper_left_test_joint'],
        finger_geometry_basis='measured_master_with_official_urdf_mimics',modeled_tool_allowance_m=.005)


def _certificate(n,front,plan,bay,measured,**kwargs):
    binding=dict(aperture_interval=vars(m.INTERVAL), assets=n._preclose_aperture_model.asset_binding,
        policy=_policy_values(n),front=np.asarray(front).tolist(),route=np.asarray(plan.route).tolist(),
        attached_corners=plan.attached_corners.tolist(),measured_start=measured['left'].tolist(),
        right_positions=[measured['joints'][name] for name in RIGHT_ARM_JOINTS],
        head_positions=[measured['joints'][name] for name in ('head_1_joint','head_2_joint')],
        bay=vars(bay),lift_m=n.lift_first_extraction_lift_m,modeled_tool_allowance_m=.005,
        context=dict(capture_clock_ns=measured['stamp_ns'],base_odom=measured['base_pose'].tolist(),
            observed_master_m=measured['fingers'][m.MASTER],reference_basis='predicted_nominal_master',
            reported_passive_joints=[]),
        software_sha256={name:hashlib.sha256(Path(importlib.import_module('erc_phase1_solution.'+name).__file__).read_bytes()).hexdigest()
                        for name in ('kinematics','shelf_cradle_geometry','lift_first_extraction')},
        interval_helper_sha256=hashlib.sha256(Path(importlib.import_module('erc_phase1_solution.aperture_interval_geometry').__file__).read_bytes()).hexdigest())
    encoded=_canonical(binding)
    return IntervalCertificate(encoded,hashlib.sha256(encoded.encode()).hexdigest(),'{}')


def state_fixture(monkeypatch):
    n=SimpleNamespace(_lock=threading.Lock(),_cancel=threading.Event(),_gripper_open_confirmed=True,
        gripper_open=.069,adaptive_start_tolerance=.001,pick_position_tolerance=.0005,
        pick_orientation_tolerance=.01,lift_first_extraction_lift_m=.005,
        carried_book_dimensions=np.asarray([.16,.02,.25]),_preclose_aperture_model=Model(),
        _shelf_cradle_geometry=SimpleNamespace(mimics={'gripper_left_test_joint':None}),
        chain=SimpleNamespace(forward=lambda q:np.asarray(q),pose_error=lambda a,b:np.zeros(6)))
    # Calling this while the final sensor lock is held would deadlock in Node.
    n._adaptive_overload_reason=lambda:pytest.fail('must read latch without locking recursively')
    n._check_lift_collision_context=ManipulationNode._check_lift_collision_context
    events=[];n._publish_status=lambda event,**fields:events.append((event,fields))
    n.current=_sample();reference=deepcopy(n.current)
    n._lift_first_measurements=lambda ref=None:deepcopy(n.current)
    front=np.array([.7,0.,1.58]);bay=SimpleNamespace(test_scope='synthetic')
    plan=SimpleNamespace(route=tuple(np.zeros(8) for _ in range(5)),attached_corners=np.zeros((8,3)))
    monkeypatch.setattr(m,'_model',lambda node:node._preclose_aperture_model)
    monkeypatch.setattr(m,'certify_predicted_route',_certificate)
    m.invalidate(n)
    return n,front,np.zeros(8),plan,bay,reference,events


def ready(monkeypatch):
    values=state_fixture(monkeypatch);n,front,grasp,plan,bay,reference,_=values
    state=m.prepare(n,front,grasp,plan,bay,reference)
    n.current['fingers'][m.MASTER]=.0183;n.current['joints'][m.MASTER]=.0183
    n.current['stamp_ns']+=2_000_000
    checked=m.reuse(n,front,grasp,plan,bay,reference)
    n._held_book_corners=plan.attached_corners.copy()
    return values,state,checked


def test_actual_open_is_separate_from_predicted_and_reuse_returns_actual_closed(monkeypatch):
    values,state,checked=ready(monkeypatch)
    n,_,_,_,_,_,events=values
    assert state.binding()['context']['observed_master_m']==.069
    assert state.binding()['aperture_interval']['reference_m']==.01825
    assert checked['fingers']=={m.MASTER:.0183}
    assert checked['geometry_context']['stamp_ns']==1_000_000_000
    assert checked['stamp_ns']==1_002_000_000
    assert events[-1][1]['original_geometry_reference_basis']=='predicted_nominal_master'
    assert events[-1][1]['drift_is_proved_displacement_envelope'] is False
    assert events[-1][1]['pressure_or_freshness_verified'] is False
    with pytest.raises(RuntimeError,match='reused'):m.reuse(n,*values[1:6])


@pytest.mark.parametrize('change', ['absent','left','right','head','base','clock','passive','basis','range','route',
    'attachment','front','bay','policy','orientation','lift','original_reference','allowance','cancel','force'])
def test_changed_context_rejects_without_recomputing(monkeypatch,change):
    values,state,checked=ready(monkeypatch);n,front,grasp,plan,bay,reference,_=values
    calls=[];monkeypatch.setattr(m,'certify_predicted_route',lambda *a,**k:calls.append(True))
    if change=='absent':n._preclose_aperture_state=None
    elif change=='left':n.current['left'][3]+=.000201
    elif change=='right':n.current['joints'][RIGHT_ARM_JOINTS[1]]+=.00101
    elif change=='head':n.current['joints']['head_2_joint']+=.00101
    elif change=='base':n.current['base_pose'][0]+=.00201
    elif change=='clock':n.current['stamp_ns']=999_999_999
    elif change=='passive':n.current['reported_passive_joints']=['gripper_left_test_joint']
    elif change=='basis':n.current['finger_geometry_basis']='measured_master_and_all_passive_joints'
    elif change=='range':n.current['fingers'][m.MASTER]=.019000001
    elif change=='route':plan.route[0][2]+=.000001
    elif change=='attachment':plan.attached_corners[0,0]+=.000001
    elif change=='front':front[1]+=.000001
    elif change=='bay':bay.test_scope='changed'
    elif change=='policy':n.carried_book_dimensions[0]+=.000001
    elif change=='orientation':n.pick_orientation_tolerance=.02
    elif change=='lift':n.lift_first_extraction_lift_m=.004
    elif change=='original_reference':reference['joints']['head_1_joint']+=.000001
    elif change=='allowance':n.current['modeled_tool_allowance_m']=0.
    elif change=='cancel':n._cancel.set()
    else:n._adaptive_overload_latched='force_overload'
    with pytest.raises((ValueError,RuntimeError)):m.check_after_probe(n,checked,n.current)
    assert calls==[]


def test_engineering_drift_remains_bound_to_initial_capture_not_previous_admission(monkeypatch):
    values,state,checked=ready(monkeypatch);n=values[0]
    n.current['left'][2]=.00015
    result=m.check_after_probe(n,checked,n.current)
    assert result['exact_arm_context_match'] is False
    assert result['drift_is_proved_displacement_envelope'] is False
    n.current['left'][2]=.00025
    with pytest.raises(ValueError,match='joint-vector drift'):m.check_after_probe(n,checked,n.current)


@pytest.mark.parametrize('change',['open','passive','reference','generation','stale'])
def test_capture_rejects_changed_state_after_geometry(monkeypatch,change):
    values=state_fixture(monkeypatch);n,front,grasp,plan,bay,reference,_=values
    def certificate(*args,**kwargs):
        result=_certificate(*args,**kwargs)
        if change=='open':n._gripper_open_confirmed=False
        elif change=='passive':n.current['reported_passive_joints']=['gripper_left_test_joint']
        elif change=='reference':reference['joints']['head_1_joint']+=.000001
        elif change=='generation':m.invalidate(n)
        else:n._lift_first_measurements=lambda *a:(_ for _ in ()).throw(RuntimeError('stale measurement'))
        return result
    monkeypatch.setattr(m,'certify_predicted_route',certificate)
    with pytest.raises((RuntimeError,ValueError)):m.prepare(n,front,grasp,plan,bay,reference)
    assert n._preclose_aperture_state is None


def test_open_invalidates_before_failed_actuation(monkeypatch):
    values,_,_=ready(monkeypatch);n=values[0]
    # The production open's next command is an intentional failing sentinel.
    n.preclose_aperture_geometry_enabled=True
    def fail_open(q, *, respect_cancel):
        assert respect_cancel is True
        return False
    n._command_gripper=fail_open
    n._target_robot_contact_latched=False
    result=ManipulationNode._open_gripper(n)
    assert result is False
    assert n._preclose_aperture_state is None
    assert n._preclose_aperture_checked is None


def test_installed_asset_paths_resolve_explicit_package_shares(monkeypatch,tmp_path):
    import ament_index_python.packages as packages
    values=state_fixture(monkeypatch);n=values[0]
    n._preclose_aperture_model=None;seen=[]
    monkeypatch.setattr(packages,'get_package_share_directory',lambda name:str(tmp_path/name))
    monkeypatch.setattr(m,'OfficialNominalModel',lambda *args:seen.append(args) or Model())
    # Call the original resolver without reloading classes used by other tests.
    _PRODUCTION_MODEL_RESOLVER(n)
    assert seen[0][0]==tmp_path/'erc_description/urdf/tiago_pro.urdf'
    assert seen[0][2]==tmp_path/'erc_phase1_solution/config/aperture_interval_official_assets.json'


def test_disabled_pick_does_not_import_or_call_optional_geometry(monkeypatch):
    monkeypatch.setattr(m,'prepare',lambda *a:pytest.fail('disabled interval preparation'))
    result=_mocked_pick_trace([.7,0.,1.58])
    assert result['succeeded'] and result['arm_moves']==[10,11,12,11]


@pytest.mark.parametrize('failure',[None,'prepare','close','reuse','after_probe'])
def test_real_pick_order_no_postclose_recompute_and_fatal_reuse(monkeypatch,failure):
    order=[]
    def setup_extra(node,seen):
        node.preclose_aperture_geometry_enabled=True;node.fine_gripper_close_enabled=True
        node._recover_ambiguous_grasp=lambda *a:order.append('empty_recovery') or False
        previous=node._close_for_grasp
        node._close_for_grasp=lambda:order.append('close') or ((False,0.,False,False,False) if failure=='close' else previous())
        original_move=node._move_arm_solution
        node._move_arm_solution=lambda q,t:order.append('arm:'+str(int(q[1]))) or original_move(q,t)
        node._recheck_lift_first_geometry=lambda *a:pytest.fail('post-close mesh recompute')
        sample=node._lift_first_measurements()
        def prepare(*args):
            order.append('prepare')
            if failure=='prepare':raise RuntimeError('geometry rejected')
            node._preclose_aperture_state=object()
        def reuse(*args):
            order.append('reuse')
            if failure=='reuse':raise RuntimeError('certificate rejected')
            return sample
        def after(*args):
            order.append('after_probe')
            if failure=='after_probe':raise RuntimeError('late drift')
        monkeypatch.setattr(m,'prepare',prepare);monkeypatch.setattr(m,'reuse',reuse)
        monkeypatch.setattr(m,'check_after_probe',after)
    seen,setup=_enabled(monkeypatch,configure=setup_extra)
    if failure in ('reuse','after_probe'):
        with pytest.raises(RuntimeError,match='pick_recovery_failed'):_mocked_pick_trace([.7,0.,1.58],configure_node=setup)
        assert seen['node']._held_book_corners is not None
    else:
        result=_mocked_pick_trace([.7,0.,1.58],configure_node=setup)
        assert result['succeeded'] is (failure is None)
    assert order.index('prepare')>order.index('arm:12')
    if failure=='prepare':assert 'close' not in order
    elif failure=='close':assert 'reuse' not in order
    else:assert order.index('reuse')>order.index('close')
    if failure:assert 'arm:40' not in order
    else:assert order.index('arm:40')>order.index('after_probe')
    if failure in ('prepare','close'):assert seen['node']._preclose_aperture_state is None


def _attach_real_pressure_gate(monkeypatch):
    gate,n,now,wall,sent,events,joint,send=pressure_fixture(monkeypatch)
    values,state,checked=ready(monkeypatch)
    synthetic=values[0]
    # Bind a synthetic geometric certificate to the actual callback fixture's
    # sensor values, without pretending that these meshes were validated.
    measured=_sample(.069);measured['left']=np.asarray([n.joints[x] for x in IK_JOINTS])
    measured['joints']=dict(n.joints);measured['joints'][m.MASTER]=.069
    measured['base_pose']=np.asarray(n._staging_odom['pose']);measured['stamp_ns']=now.nanoseconds
    synthetic.current=measured
    reference=deepcopy(measured);front,plan,bay=values[1],values[3],values[4]
    cert=_certificate(synthetic,front,plan,bay,measured)
    state=replace(state,certificate=cert,grasp=tuple(measured['left']),reference_source=reference,
                  original_reference_signature=m._reference_signature(reference))
    for name in ('chain','pick_position_tolerance','pick_orientation_tolerance','lift_first_extraction_lift_m',
                 'carried_book_dimensions','_shelf_cradle_geometry'):
        setattr(n,name,getattr(synthetic,name))
    n._preclose_aperture_generation=state.generation;n._preclose_aperture_state=state
    n._preclose_aperture_checked=state;n._preclose_aperture_dispatched=False
    n._held_book_corners=plan.attached_corners.copy()
    n._adaptive_overload_reason=lambda:pytest.fail('recursive lock getter')
    gate.checked['preclose_interval_admission']=m.FinalIntervalAdmission(state)
    return gate,n,now,wall,sent,events,joint,send


def test_real_callback_gate_consumes_interval_under_nonreentrant_lock_once(monkeypatch):
    gate,n,now,wall,sent,events,joint,send=_attach_real_pressure_gate(monkeypatch)
    assert gate.send(send)=='accepted-future'
    assert n._preclose_aperture_dispatched and len(sent)==1
    second=lift_pressure_gate.LiftPressureGate(n,gate.checked,gate.reference)
    with pytest.raises(lift_pressure_gate.LiftPressureRejected,match='consumed'):second.send(send)
    assert len(sent)==1


@pytest.mark.parametrize('change',['range','original_left_drift','attachment','reset'])
def test_late_callback_invalidates_final_interval_without_arm_publish(monkeypatch,change):
    gate,n,now,wall,sent,events,joint,send=_attach_real_pressure_gate(monkeypatch)
    def update():
        if change=='range':
            # Within the existing10um finger-drift tolerance, across fixed interval.
            n.joints[m.MASTER]=.019000001;n._on_joint_state(joint())
        elif change=='original_left_drift':n.joints[IK_JOINTS[2]]+=.00010
        elif change=='attachment':n._held_book_corners[0,0]=.000001
        else:m.invalidate(n)
    if change=='range':
        n.joints[m.MASTER]=.018999
        gate.checked['fingers'][m.MASTER]=.018999
        n._on_joint_state(joint())
    elif change=='original_left_drift':
        n.joints[IK_JOINTS[2]]=.00015
        gate.checked['left'][2]=.00015
        n._on_joint_state(joint())
    wall.hook=update
    with pytest.raises(lift_pressure_gate.LiftPressureRejected):gate.send(send)
    assert not sent


def test_default_parameter_and_installed_config_stay_disabled():
    node=object.__new__(ManipulationNode);values={}
    node.declare_parameter=lambda name,value:values.setdefault(name,value)
    node._declare_parameters()
    assert values['preclose_aperture_geometry_enabled'] is False
    import yaml
    root=Path(__file__).parents[1]
    config=yaml.safe_load((root/'config/solution.yaml').read_text())
    assert config['erc_manipulation']['ros__parameters']['preclose_aperture_geometry_enabled'] is False
    assert (root/'config/aperture_interval_official_assets.json').is_file()
    assert "glob('config/*')" in (root/'setup.py').read_text()


@pytest.mark.parametrize('missing',['lift','fine'])
def test_enabled_option_requires_both_existing_profiles_before_commands(monkeypatch,missing):
    def configure(node,seen):
        node.preclose_aperture_geometry_enabled=True
        node.fine_gripper_close_enabled=missing!='fine'
        node.lift_first_extraction_enabled=missing!='lift'
        node._move_torso=lambda *a:pytest.fail('invalid profile moved')
    _,setup=_enabled(monkeypatch,configure=configure)
    with pytest.raises(RuntimeError,match='requires lift-first and fine'):
        _mocked_pick_trace([.7,0.,1.58],configure_node=setup)


def test_new_attempt_clears_previous_certificate_before_perception_failure(monkeypatch):
    def configure(node,seen):
        node.preclose_aperture_geometry_enabled=True;node.fine_gripper_close_enabled=True
        node._preclose_aperture_state=object();node._preclose_aperture_checked=object()
    seen,setup=_enabled(monkeypatch,configure=configure,decode_error=True)
    with pytest.raises(ValueError,match='registration absent'):
        _mocked_pick_trace([.7,0.,1.58],configure_node=setup)
    assert seen['node']._preclose_aperture_state is None
    assert seen['node']._preclose_aperture_checked is None
