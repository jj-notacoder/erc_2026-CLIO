"""Optional lift wiring against production pickup and measured-state methods."""
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip('rclpy')
from erc_phase1_solution import manipulation_node as manipulation
from erc_phase1_solution import lift_first_extraction as lift
from erc_phase1_solution import shelf_bay_context as decoder
from test_shutdown import _mocked_pick_trace


def _enabled(monkeypatch, *, configure=None, decode_error=False):
    seen = {}
    bay = object()
    class SentinelPressureGate:
        """Order-only adapter: actual pressure/feedback has separate ROS cases."""
        def __init__(self, node, checked, reference):
            seen['pressure_gate_arguments'] = (node, checked, reference)
            seen['pressure_gate_calls'] = 0
        def send(self, publish):
            assert seen['pressure_gate_calls'] == 0
            seen['pressure_gate_calls'] += 1
            seen['pressure_gate_order'] = ['admitted']
            result = publish()
            seen['pressure_gate_order'].append('published')
            return result
    monkeypatch.setattr(manipulation, 'LiftPressureGate', SentinelPressureGate)
    def decode(node, payload):
        seen['payload'] = payload
        if decode_error: raise ValueError('registration absent')
        return bay
    monkeypatch.setattr(decoder, 'decode_bay_context', decode)
    def planner(node, front, grasp, original, **kwargs):
        seen['original'] = [q.copy() for q in original]
        seen['grasp'] = grasp.copy()
        seen['planner_kwargs'] = kwargs
        route = (np.full(8, 40.), np.full(8, 41.))
        result = lift.LiftFirstPlan(route, route[-1].copy(), np.zeros((8,3)),
            {'full_shelf_collision_certificate':False, 'deferred_carry_regenerated':False})
        seen['lift'] = result
        return result
    monkeypatch.setattr(lift, 'plan_lift_first_extraction', planner)
    def setup(node):
        seen['node'] = node
        node._lock = threading.RLock()
        node._staging_odom = {'pose':[0.,0.,0.]}
        node.lift_first_extraction_enabled = True
        node.lift_first_extraction_lift_m = .005
        node.carried_book_dimensions = np.asarray([.16,.02,.25])
        node._shelf_cradle_geometry = object()
        parked = dict(zip((*manipulation.RIGHT_ARM_JOINTS, *manipulation.HEAD_JOINTS),
                          (*manipulation.RIGHT_HOME, 0., -.28)))
        sample = dict(left=np.full(8,12.),base_pose=np.zeros(3),
                      fingers={'gripper_left_finger_joint':.018},joints=parked,
                      stamp_ns=1_000_000_000,
                      geometry_context=dict(
                          right_positions=tuple(parked[n] for n in manipulation.RIGHT_ARM_JOINTS),
                          head_positions=tuple(parked[n] for n in manipulation.HEAD_JOINTS),
                          stamp_ns=1_000_000_000))
        node._lift_first_measurements = lambda reference=None: sample
        def measured(*args):
            seen['recheck'] = args
            seen['held_during_recheck'] = node._held_book_corners is not None
            return sample
        node._recheck_lift_first_geometry = measured
        if configure: configure(node,seen)
    return seen,setup


def test_disabled_option_keeps_original_approach_and_withdrawal(monkeypatch):
    monkeypatch.setattr(decoder,'decode_bay_context',lambda *a: pytest.fail('disabled decoder'))
    result=_mocked_pick_trace([.70,0.,1.58])
    assert result['succeeded']
    assert result['arm_moves'] == [10,11,12,11]
    assert result['arm_durations'] == [2.8,.65,.65,5.8]
    assert not any(name.startswith('lift_first') for name,_ in result['status'])


def test_enabled_plans_carry_from_raised_terminal_and_preserves_empty_approach(monkeypatch):
    seen,setup=_enabled(monkeypatch)
    result=_mocked_pick_trace([.70,0.,1.58],configure_node=setup)
    assert result['succeeded']
    assert [q[1] for q in seen['original']] == [11]
    assert seen['grasp'][1] == 12
    assert result['plan']['clearance'][1] == 41
    assert result['carried_staging'][1] == 41
    assert result['arm_moves'] == [10,11,12,40,41]
    assert result['arm_durations'] == [2.8,.65,.65,1.,5.8]
    assert seen['held_during_recheck']
    assert seen['pressure_gate_calls'] == 1
    assert seen['pressure_gate_order'] == ['admitted', 'published']
    assert seen['pressure_gate_arguments'][1]['geometry_context']['head_positions'] == (0., -.28)
    assert [x[1] for x in result['fresh_probes']] == ['before_initial_shelf_lift','initial_shelf_lift']
    event=next(f for e,f in result['status'] if e=='lift_first_extraction_planned')
    assert event['deferred_carry_regenerated'] is True
    assert event['full_shelf_collision_certificate'] is False
    assert seen['node']._cached_post_retreat_plan['sentinel']=='supported_post_retreat_return'


def test_registration_rejected_before_empty_actuation(monkeypatch):
    seen,setup=_enabled(monkeypatch,decode_error=True)
    moves=[]
    def configure(node):
        setup(node)
        node._move_arm_solution=lambda *a: moves.append('arm')
        node._move_torso=lambda *a: moves.append('torso')
        node._open_gripper=lambda: moves.append('open')
    with pytest.raises(ValueError,match='registration absent'):
        _mocked_pick_trace([.70,0.,1.58],configure_node=configure)
    assert moves==[]


def test_measured_geometry_rejection_halts_closed_without_loaded_dispatch(monkeypatch):
    def configure(node,seen):
        def reject(*args): raise ValueError('actual passive finger collision')
        node._recheck_lift_first_geometry=reject
        seen['moves']=[]
        original=node._move_arm_solution
        node._move_arm_solution=lambda q,t: seen['moves'].append(q[1]) or original(q,t)
        node._recover_closed_pick=lambda **kw: pytest.fail('unchecked recovery')
    seen,setup=_enabled(monkeypatch,configure=configure)
    with pytest.raises(RuntimeError,match='pick_recovery_failed'):
        _mocked_pick_trace([.70,0.,1.58],configure_node=setup)
    assert seen['moves']==[10,11,12]
    assert seen['node']._held_book_corners is not None


@pytest.mark.parametrize('failed_phase,expected_moves,remaining',[
    ('before_initial_shelf_lift',[10,11,12],[40,41]),
    ('initial_shelf_lift',[10,11,12,40],[41]),
])
def test_fresh_retention_loss_stops_before_unchecked_next_leg(monkeypatch,failed_phase,expected_moves,remaining):
    def configure(node,seen):
        node._fresh_retention_probe=lambda command,phase,**kw: phase!=failed_phase
    seen,setup=_enabled(monkeypatch,configure=configure)
    result=_mocked_pick_trace([.70,0.,1.58],configure_node=setup)
    assert not result['succeeded']
    assert result['arm_moves']==expected_moves
    recovery=result['closed_recoveries'][0]
    assert recovery['cause']=='contact_lost'
    assert [q[1] for q,_,_ in recovery['remaining_carried_legs']]==remaining


def test_current_pick_payload_is_forwarded_by_command_dispatch():
    node=object.__new__(manipulation.ManipulationNode)
    node._lock=threading.RLock();node._cancel=threading.Event();node.dry_run=False
    node.book_row_tilts=[];node._publish_status=lambda *a,**kw:None
    calls=[];node._pick=lambda payload=None:calls.append(payload) or True
    payload={'event':'pick','shelf_bay_context':{'source':'onboard_marker_cloud'}}
    node._run_command('pick',payload)
    node._run_command('pick')
    assert calls==[payload,None]


def _measured_node():
    node=object.__new__(manipulation.ManipulationNode)
    node._lock=threading.RLock()
    node._shelf_cradle_geometry=SimpleNamespace(chains={'tip':SimpleNamespace(active_names=(
        *manipulation.IK_JOINTS,'gripper_left_finger_joint','gripper_left_inner_knuckle_joint'))})
    names=(*manipulation.IK_JOINTS,*manipulation.RIGHT_ARM_JOINTS,*manipulation.HEAD_JOINTS,
           'gripper_left_finger_joint','gripper_left_inner_knuckle_joint')
    node.joints={n:0. for n in names}
    node.joints['gripper_left_finger_joint']=.0183
    node.joints['gripper_left_inner_knuckle_joint']=.044
    node._joint_stamps_ns={n:1_000_000_000 for n in names}
    node._staging_odom=dict(stamp_ns=1_000_000_000,pose=[.1,.2,.3],linear_speed=0.,angular_speed=0.)
    node.get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=1_010_000_000))
    return node


def test_measured_geometry_uses_all_current_right_head_passive_fingers():
    node=_measured_node();data=node._lift_first_measurements()
    assert data['fingers']=={'gripper_left_finger_joint':.0183,'gripper_left_inner_knuckle_joint':.044}
    assert data['base_pose']==pytest.approx([.1,.2,.3])
    node.joints['head_1_joint']=.01
    with pytest.raises(RuntimeError,match='collision geometry moved'):
        node._lift_first_measurements(data)


@pytest.mark.parametrize('name',['arm_right_4_joint','head_2_joint','gripper_left_inner_knuckle_joint','arm_left_3_joint'])
def test_stale_measured_geometry_fails_without_home_fallback(name):
    node=_measured_node();node._joint_stamps_ns[name]=0
    with pytest.raises(RuntimeError,match='measurement stale'):
        node._lift_first_measurements()


@pytest.mark.parametrize('change',['translation','yaw','moving','stale_odom','nan'])
def test_base_registration_cannot_survive_movement_or_missing_measurements(change):
    node=_measured_node();original=node._lift_first_measurements()
    if change=='translation':node._staging_odom['pose'][0]+=.003
    elif change=='yaw':node._staging_odom['pose'][2]+=.006
    elif change=='moving':node._staging_odom['linear_speed']=.006
    elif change=='stale_odom':node._staging_odom['stamp_ns']=0
    else:node._staging_odom['pose'][2]=float('nan')
    with pytest.raises(RuntimeError):node._lift_first_measurements(original)


def test_parameters_default_off_with_true_five_mm_option():
    node=object.__new__(manipulation.ManipulationNode);values={}
    node.declare_parameter=lambda name,value:values.setdefault(name,value)
    node._declare_parameters()
    assert values['lift_first_extraction_enabled'] is False
    assert values['lift_first_extraction_lift_m']==.005


@pytest.mark.parametrize('failure',['stale_base','arm_drift','finger_drift'])
def test_late_geometry_rejection_is_fatal_closed_and_never_dispatches_lift(monkeypatch,failure):
    def configure(node,seen):
        original=node._lift_first_measurements
        seen['probe_finished']=False
        def probe(command,phase,**kwargs):
            if phase=='before_initial_shelf_lift':seen['probe_finished']=True
            return True
        node._fresh_retention_probe=probe
        def measurements(reference=None):
            data=original(reference)
            if not seen['probe_finished']:return data
            if failure=='stale_base':raise RuntimeError('lift-first odometry is stale')
            data={**data,'left':data['left'].copy(),'fingers':dict(data['fingers'])}
            if failure=='arm_drift':data['left'][2]+=.001
            else:data['fingers']['gripper_left_finger_joint']+=.00002
            return data
        node._lift_first_measurements=measurements
        seen['moves']=[];original_move=node._move_arm_solution
        node._move_arm_solution=lambda q,t:seen['moves'].append(q[1]) or original_move(q,t)
        seen['statuses']=[];original_status=node._publish_status
        node._publish_status=lambda event,**fields:(seen['statuses'].append((event,fields)),original_status(event,**fields))
        node._recover_closed_pick=lambda **kw:pytest.fail('unchecked recovery')
    seen,setup=_enabled(monkeypatch,configure=configure)
    with pytest.raises(RuntimeError,match='pick_recovery_failed'):
        _mocked_pick_trace([.70,0.,1.58],configure_node=setup)
    assert seen['probe_finished']
    assert seen['moves']==[10,11,12]
    assert seen['node']._held_book_corners is not None
    event=next(f for e,f in seen['statuses'] if e=='lift_first_preflight_rejected')
    assert event['phase']=='after_contact_probe'
    assert event['retained_stop'] and event['recovery_halted']


# Exact actuated state-interface names in the official URDF, independently seen
# in Run13 preparation /joint_states (no mission was dispatched).
_OFFICIAL_ACTUATED_NAMES=(
    'torso_lift_joint','head_1_joint','head_2_joint',
    *(f'arm_left_{i}_joint' for i in range(1,8)),
    'gripper_left_finger_joint',*(f'arm_right_{i}_joint' for i in range(1,8)),
    'gripper_right_finger_joint','wheel_front_right_joint','wheel_front_left_joint',
    'wheel_rear_right_joint','wheel_rear_left_joint')
_OFFICIAL_PASSIVE_NAMES=(
    'gripper_left_outer_finger_right_joint','gripper_left_inner_finger_right_joint',
    'gripper_left_fingertip_right_joint','gripper_left_finger_right_joint',
    'gripper_left_outer_finger_left_joint','gripper_left_inner_finger_left_joint',
    'gripper_left_fingertip_left_joint')


def _official_actuated_node():
    node=_measured_node()
    node._shelf_cradle_geometry=SimpleNamespace(chains={'all':SimpleNamespace(active_names=(
        *manipulation.IK_JOINTS,'gripper_left_finger_joint',*_OFFICIAL_PASSIVE_NAMES))})
    node.joints={name:0. for name in _OFFICIAL_ACTUATED_NAMES}
    node.joints['gripper_left_finger_joint']=.0183
    node._joint_stamps_ns={name:1_000_000_000 for name in _OFFICIAL_ACTUATED_NAMES}
    return node


def test_actual_official_23_joint_layout_uses_measured_master_and_named_modeled_mimics():
    node=_official_actuated_node()
    assert len(node.joints)==23
    sample=node._lift_first_measurements()
    assert sample['fingers']=={'gripper_left_finger_joint':.0183}
    assert set(sample['modeled_finger_joints'])==set(_OFFICIAL_PASSIVE_NAMES)
    assert sample['finger_geometry_basis']=='measured_master_with_official_urdf_mimics'
    assert sample['modeled_tool_allowance_m']==.005
    assert sample['reported_passive_joints']==[]


@pytest.mark.parametrize('which',['master','left','right','head'])
def test_nominal_mimic_branch_still_requires_available_actuated_feedback(which):
    node=_official_actuated_node()
    name={'master':'gripper_left_finger_joint','left':'arm_left_2_joint',
          'right':'arm_right_3_joint','head':'head_1_joint'}[which]
    del node.joints[name]
    with pytest.raises(RuntimeError,match='measurement stale'):node._lift_first_measurements()


def test_partial_fresh_passive_set_uses_consistent_modeled_only_linkage():
    node=_official_actuated_node();name=_OFFICIAL_PASSIVE_NAMES[0]
    node.joints[name]=.1;node._joint_stamps_ns[name]=1_000_000_000
    sample=node._lift_first_measurements()
    assert sample['reported_passive_joints']==[name]
    assert sample['fingers']=={'gripper_left_finger_joint':.0183}
    assert len(sample['modeled_finger_joints'])==7


@pytest.mark.parametrize('invalid',['stale','nonfinite'])
def test_reported_invalid_passive_feedback_is_not_hidden_by_nominal_fallback(invalid):
    node=_official_actuated_node();name=_OFFICIAL_PASSIVE_NAMES[0]
    node.joints[name]=float('nan') if invalid=='nonfinite' else .1
    node._joint_stamps_ns[name]=1_000_000_000 if invalid=='nonfinite' else 0
    with pytest.raises(RuntimeError,match='measurement stale'):node._lift_first_measurements()


def test_complete_fresh_passive_set_is_measured_and_no_model_allowance_is_claimed():
    node=_official_actuated_node()
    for name in _OFFICIAL_PASSIVE_NAMES:
        node.joints[name]=.1;node._joint_stamps_ns[name]=1_000_000_000
    sample=node._lift_first_measurements()
    assert len(sample['fingers'])==8
    assert sample['modeled_finger_joints']==[]
    assert sample['finger_geometry_basis']=='measured_master_and_all_passive_joints'
    assert sample['modeled_tool_allowance_m']==0.


def test_post_close_recheck_labels_nominal_passives_and_passes_tool_only_allowance(monkeypatch):
    node=_official_actuated_node();seen={};statuses=[]
    node.chain=SimpleNamespace(forward=lambda q:np.eye(4),pose_error=lambda a,b:np.zeros(6))
    node.pick_position_tolerance=.0005;node.pick_orientation_tolerance=.01
    node.lift_first_extraction_lift_m=.005
    node._publish_status=lambda event,**fields:statuses.append((event,fields))
    def validate(*args,**kwargs):
        seen.update(kwargs)
        return {'supplied_finger_positions_checked':True,
                'modeled_tool_allowance_m':kwargs['modeled_tool_allowance_m']}
    monkeypatch.setattr(lift,'validate_lift_first_route',validate)
    reference=node._lift_first_measurements()
    plan=lift.LiftFirstPlan((np.zeros(8),),np.zeros(8),np.zeros((8,3)),{})
    node._recheck_lift_first_geometry(np.zeros(3),np.zeros(8),plan,object(),reference)
    assert seen['finger_positions']=={'gripper_left_finger_joint':.0183}
    assert seen['modeled_tool_allowance_m']==.005
    fields=statuses[-1][1]
    assert fields['measured_finger_joints']==['gripper_left_finger_joint']
    assert set(fields['modeled_finger_joints'])==set(_OFFICIAL_PASSIVE_NAMES)
    assert fields['finger_geometry_basis']=='measured_master_with_official_urdf_mimics'
    assert 'actual_finger_geometry_checked' not in fields
