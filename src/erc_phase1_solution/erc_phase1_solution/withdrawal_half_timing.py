"""Default-off timing for four already checked normal top-row withdrawals."""
from dataclasses import dataclass
import copy
import math
import struct

import numpy as np

from .arm_velocity_admission import ArmVelocityAdmissionRejected
from .completed_torso_hold import TorsoHoldCancellation
from .lift_first_extraction import LiftFirstPlan
from .lift_pressure_gate import LiftPressureGate
from .motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS
from .withdrawal_timing import withdrawal_leg_speed_scales

HEAD = ('head_1_joint', 'head_2_joint')
MODEL_NAMES = ('chain', 'right_chain', 'head_chain', 'carried_collision_meshes', '_shelf_cradle_geometry')


def checked_enabled(value):
    if type(value) is not bool:
        raise ValueError('withdrawal_half_timing_enabled must be Boolean')
    return value


def _input_fraction(node):
    quarter = getattr(node, 'withdrawal_quarter_timing_enabled', False)
    if type(quarter) is not bool:
        raise ValueError('withdrawal_quarter_timing_enabled must be Boolean')
    return .25 if quarter else .5


def _fail(reason):
    raise ArmVelocityAdmissionRejected('withdrawal timing: ' + reason)


def _bits(values):
    values = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in values):
        _fail('nonfinite route')
    return struct.pack('!'+str(len(values))+'d', *values)


def _route(legs):
    if len(legs) != 5:
        _fail('requires the original five-leg prefix')
    expected = ((1., 'initial_shelf_lift'), *((5.8, 'extraction'),)*4)
    key=[]
    for leg, (duration, phase) in zip(legs, expected):
        if (len(leg) != 3 or len(leg[0]) != 8 or type(leg[1]) is bool
                or leg[1] != duration or leg[2] != phase):
            _fail('original route phases or durations changed')
        key.append((_bits(leg[0]), duration, phase))
    return tuple(key)


@dataclass(frozen=True)
class WithdrawalTiming:
    node: object
    route_key: tuple
    targets: tuple
    pressure_gate: LiftPressureGate
    reference: dict
    model: str
    corners: object
    corners_bytes: bytes
    model_objects: tuple
    contact_epoch: int
    cancel: TorsoHoldCancellation
    cancel_generation: int
    input_fraction: float = .5


@dataclass(frozen=True)
class WithdrawalLeg:
    owner: WithdrawalTiming
    index: int


def _scope_locked(node, timing, *, after_lift):
    if type(timing) is not WithdrawalTiming or timing.node is not node:
        _fail('foreign route owner')
    if (not checked_enabled(getattr(node, 'withdrawal_half_timing_enabled', False))
            or _input_fraction(node) != timing.input_fraction
            or node._cancel is not timing.cancel
            or timing.cancel.snapshot() != (timing.cancel_generation, False)
            or getattr(node, 'withdrawal_speed_scale', 1.) != 3.
            or getattr(node, 'additional_arm_time_scale', 1.) != 2.
            or getattr(node, 'lift_first_extraction_lift_m', None) != .020
            or getattr(node, '_goal_handles', ()) or getattr(node, '_pending_retained_acceptances', ())):
        _fail('configuration, cancellation or action ownership changed')
    gate = timing.pressure_gate
    if (type(gate) is not LiftPressureGate or gate.node is not node
            or gate.expected_model != timing.model or gate.used is not after_lift
            or gate.send_started is not after_lift
            or getattr(node, '_target_book_model', None) != timing.model
            or getattr(node, '_contact_epoch', None) != timing.contact_epoch + int(after_lift)
            or getattr(node, '_held_book_corners', None) is not timing.corners
            or timing.corners.tobytes() != timing.corners_bytes
            or any(getattr(node, name) is not obj for name,obj in zip(MODEL_NAMES,timing.model_objects))
            or getattr(node, '_active_place_scene_reference', None) is not None
            or getattr(node, '_gravity_supported_payload', False)
            or not getattr(node, '_payload_monitor_enabled', False)
            or getattr(node, '_gripper_open_confirmed', False)
            or getattr(node, '_payload_hazard_latched', None) is not None
            or getattr(node, '_held_grip_sensor_fault', None) is not None
            or getattr(node, '_target_robot_contact_latched', False)
            or getattr(node, '_raw_contacts_first_failure', None) is not None):
        _fail('held payload, geometry or lift/probe identity changed')
    now = node.get_clock().now().nanoseconds
    for name in (*RIGHT_ARM_JOINTS, *HEAD):
        value=node.joints.get(name,math.nan);stamp=node._joint_stamps_ns.get(name)
        if (type(stamp) is not int or not -50_000_000 <= now-stamp <= 350_000_000
                or not math.isfinite(value) or abs(value-timing.reference['joints'][name]) > .001):
            _fail('parked right/head context changed')
    odom=node._staging_odom
    pose=np.asarray(odom['pose'],dtype=float);previous=np.asarray(timing.reference['base_pose'],dtype=float)
    speeds=np.asarray([odom['linear_speed'],odom['angular_speed']],dtype=float)
    if (pose.shape!=(3,) or previous.shape!=(3,) or not np.isfinite(pose).all()
            or not np.isfinite(previous).all() or type(odom['stamp_ns']) is not int):
        _fail('invalid registered base context')
    yaw=math.atan2(math.sin(pose[2]-previous[2]),math.cos(pose[2]-previous[2]))
    if (not -50_000_000 <= now-odom['stamp_ns'] <= 350_000_000
            or pose.shape!=(3,) or not np.isfinite(pose).all() or not np.isfinite(speeds).all()
            or np.any(speeds<0) or speeds[0]>.005 or speeds[1]>.008
            or np.linalg.norm(pose[:2]-previous[:2])>.002 or abs(yaw)>.005):
        _fail('registered stationary base changed')


def normal_options(node, legs, pressure_gate, lift_plan, *, ordinary):
    if not checked_enabled(getattr(node, 'withdrawal_half_timing_enabled', False)) or ordinary is not True:
        return {}
    try:
        if (type(lift_plan) is not LiftFirstPlan
                or lift_plan.metrics.get('lift_m') != .020
                or type(pressure_gate) is not LiftPressureGate
                or type(node._cancel) is not TorsoHoldCancellation
                or len(lift_plan.route) != 5
                or any(_bits(q)!=_bits(leg[0]) for q,leg in zip(lift_plan.route,legs))):
            _fail('requires ordinary checked top-row20mm lift route')
        key=_route(legs)
        withdrawal_leg_speed_scales(legs,command='pick',speed_scale=3.,arm_speed_scale=1.,
            leg_offset=0,initial_pressure_gate=pressure_gate)
        with node._lock:
            cancel_generation,cancelled=node._cancel.snapshot()
            corners=node._held_book_corners
            if cancelled or type(corners) is not np.ndarray or corners.shape!=(8,3) or not np.isfinite(corners).all():
                _fail('invalid held attachment')
            timing=WithdrawalTiming(node,key,tuple(tuple(float(v) for v in leg[0]) for leg in legs),
                pressure_gate,copy.deepcopy(pressure_gate.reference),pressure_gate.expected_model,
                corners,corners.tobytes(),tuple(getattr(node,n) for n in MODEL_NAMES),
                node._contact_epoch,node._cancel,cancel_generation,_input_fraction(node))
            _scope_locked(node,timing,after_lift=False)
        return {'withdrawal_timing':timing}
    except (AttributeError,KeyError,TypeError,ValueError,ArmVelocityAdmissionRejected) as error:
        raise RuntimeError('pick_recovery_failed') from error


def validate_execution(node,timing,legs,command,*,leg_offset,initial_pressure_gate,
                       arm_speed_scale,withdrawal_speed_scale,fresh_retention_phases,clearance_timing):
    try:
        if (command!='pick' or leg_offset!=0 or initial_pressure_gate is not timing.pressure_gate
                or arm_speed_scale!=1. or withdrawal_speed_scale!=3.
                or tuple(fresh_retention_phases)!=('initial_shelf_lift',) or clearance_timing is not None
                or _route(legs)!=timing.route_key):
            _fail('marker cannot apply to another route or recovery')
        with node._lock:_scope_locked(node,timing,after_lift=False)
    except (AttributeError,KeyError,TypeError,ValueError,ArmVelocityAdmissionRejected) as error:
        raise RuntimeError('pick_recovery_failed') from error


def timing_input(node,timing,index,solution,duration,phase):
    try:
        if not 1<=index<=4 or phase!='extraction' or duration!=5.8 or _bits(solution)!=timing.route_key[index][0]:
            _fail('withdrawal leg changed before timing')
        node._lift_first_measurements(timing.reference)
        reason=node._payload_hazard_reason(max_age=min(.15,float(node.grasp_contact_max_age)))
        if reason is not None:_fail('payload hazard: '+str(reason))
        with node._lock:_scope_locked(node,timing,after_lift=True)
        node._publish_status('withdrawal_half_timing',command='pick',phase=phase,leg=index,
            nominal_watchdog_seconds=5.8,timing_input_seconds=5.8*timing.input_fraction,
            target_model=timing.model,contact_epoch=timing.contact_epoch+1)
        return 5.8*timing.input_fraction,WithdrawalLeg(timing,index)
    except (AttributeError,IndexError,KeyError,TypeError,ValueError,RuntimeError) as error:
        raise ArmVelocityAdmissionRejected('withdrawal timing input rejected: '+str(error)) from error


def _endpoint_locked(node,timing,index):
    now=node.get_clock().now().nanoseconds
    positions=[node.joints.get(n,math.nan) for n in IK_JOINTS]
    stamps=[node._joint_stamps_ns.get(n) for n in IK_JOINTS]
    expected=timing.targets[index]
    if (any(type(s) is not int or not -50_000_000<=now-s<=150_000_000 for s in stamps)
            or not all(math.isfinite(v) for v in positions)
            or abs(positions[0]-expected[0])>.001
            or any(abs(v-q)>.005 for v,q in zip(positions[1:],expected[1:]))):
        _fail('fresh measured endpoint unresolved')


def require_publication_locked(node,marker,goal,command,leg_offset):
    try:
        if (type(marker) is not WithdrawalLeg or command!='pick' or marker.index!=leg_offset
                or not 1<=marker.index<=4 or len(goal.trajectory.points)!=1
                or tuple(goal.trajectory.joint_names)!=tuple(IK_JOINTS[1:])
                or _bits(goal.trajectory.points[0].positions)!=_bits(marker.owner.targets[marker.index][1:])):
            _fail('serialized withdrawal or route index changed')
        _scope_locked(node,marker.owner,after_lift=True)
        _endpoint_locked(node,marker.owner,marker.index-1)
    except (AttributeError,IndexError,KeyError,TypeError,ValueError,RuntimeError) as error:
        raise ArmVelocityAdmissionRejected('withdrawal publication rejected: '+str(error)) from error


def require_endpoint(node,marker):
    try:
        timing,index=marker.owner,marker.index
        with node._lock:_scope_locked(node,timing,after_lift=True)
        if node._wait_for_retained_endpoint(timing.targets[index],command='pick',phase='extraction',leg=index) is None:
            _fail('endpoint wait did not confirm arrival')
        with node._adaptive_command_guard():
            reason=node._payload_hazard_reason(max_age=min(.15,float(node.grasp_contact_max_age)))
            if reason is not None:_fail('endpoint payload hazard: '+str(reason))
            with node._lock:
                _scope_locked(node,timing,after_lift=True)
                _endpoint_locked(node,timing,index)
        node._publish_status('withdrawal_half_endpoint_verified',command='pick',phase='extraction',leg=index,
            expected=list(timing.targets[index]),torso_tolerance=.001,arm_tolerance=.005,
            target_model=timing.model,contact_epoch=timing.contact_epoch+1,stationary_verified=False)
    except (AttributeError,IndexError,KeyError,TypeError,ValueError,RuntimeError) as error:
        raise RuntimeError('pick_recovery_failed') from error
