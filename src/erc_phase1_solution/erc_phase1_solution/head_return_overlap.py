"""Default-off head-only return overlap; immutable carry context and owned goals."""
from __future__ import annotations

import math
import re
import time
import uuid
import numpy as np
from .motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS

COMMAND = 'preposition_bin_head'
HEAD = ('head_1_joint', 'head_2_joint')
TARGET = (0., -.6)
NAMES = (*IK_JOINTS, *RIGHT_ARM_JOINTS, *HEAD)


def checked_enabled(value):
    if type(value) is not bool:
        raise ValueError('head_return_overlap_enabled must be Boolean')
    return value


def identity(payload):
    result = {key: payload.get(key) for key in ('trial_id', 'head_return_id', 'target_model')}
    if (not isinstance(result['trial_id'], str) or not re.fullmatch('[0-9a-f]{12}', result['trial_id'])
            or not isinstance(result['head_return_id'], str) or not re.fullmatch('[0-9a-f]{32}', result['head_return_id'])
            or not isinstance(result['target_model'], str) or not re.fullmatch(r'book_col_\d+_row_\d+_[A-Za-z0-9_]+', result['target_model'])):
        raise RuntimeError('head_return_invalid_identity')
    return result


def snapshot_locked(node, *, reference=None, moving_head=False, stopped=False, target=False):
    """Caller owns the sensor lock; no geometry, publisher or nested lock here."""
    now = int(node.get_clock().now().nanoseconds)
    joints = dict(node.joints)
    velocities = dict(node._joint_velocities)
    stamps = dict(node._joint_stamps_ns)
    attached = getattr(node, '_held_book_corners', None)
    if (node._cancel.is_set() or attached is None or getattr(node, '_target_book_model', None) is None
            or getattr(node, '_payload_hazard_latched', None) is not None
            or getattr(node, '_held_grip_sensor_fault', None) is not None
            or getattr(node, '_target_robot_contact_latched', False)):
        raise RuntimeError('head_return_payload_or_cancel_interlock')
    attached = np.asarray(attached, dtype=float).copy()
    if attached.shape != (8, 3) or not np.isfinite(attached).all():
        raise RuntimeError('head_return_invalid_attachment')
    for name in NAMES:
        value, velocity, stamp = joints.get(name), velocities.get(name), stamps.get(name)
        if (not isinstance(stamp, int) or value is None or velocity is None
                or not math.isfinite(value) or not math.isfinite(velocity)
                or not -50_000_000 <= now-stamp <= 150_000_000):
            raise RuntimeError('head_return_stale_joint:' + name)
        limit = 1e-5 if name == 'torso_lift_joint' else 1e-4
        if not (moving_head and name in HEAD) and abs(velocity) > limit:
            raise RuntimeError('head_return_parked_joint_moving:' + name)
    if reference is not None:
        if (now < reference['evaluated_ros_ns']
                or node._target_book_model != reference['target_model']
                or int(getattr(node, '_contact_epoch', 0)) != reference['contact_epoch']
                or not np.array_equal(attached, reference['attached'])):
            raise RuntimeError('head_return_carry_identity_changed')
        for name in NAMES:
            if name in HEAD and moving_head:
                index = HEAD.index(name)
                lo, hi = sorted((reference['joints'][name], TARGET[index]))
                if not lo-1e-6 <= joints[name] <= hi+1e-6:
                    raise RuntimeError('head_return_head_left_checked_interval')
            elif (name not in HEAD or not target) and abs(joints[name]-reference['joints'][name]) > 1e-6:
                raise RuntimeError('head_return_parked_joint_drift:' + name)
    if target and any(abs(joints[n]-v) > 1e-6 for n,v in zip(HEAD,TARGET)):
        raise RuntimeError('head_return_target_not_measured')
    odom = getattr(node, '_staging_odom', None)
    if stopped:
        if (not isinstance(odom, dict) or not -50_000_000 <= now-int(odom.get('stamp_ns', -10**18)) <= 150_000_000
                or not np.isfinite([*odom['pose'],odom['linear_speed'],odom['angular_speed']]).all()
                or odom['linear_speed'] > .005 or odom['angular_speed'] > .008):
            raise RuntimeError('head_return_base_not_measured_stopped')
        if reference is not None and reference.get('base_must_match'):
            old = reference['odom']['pose'];new = odom['pose']
            angle = math.atan2(math.sin(new[2]-old[2]), math.cos(new[2]-old[2]))
            if math.hypot(new[0]-old[0],new[1]-old[1]) > .002 or abs(angle) > .005:
                raise RuntimeError('head_return_base_moved_before_head_acceptance')
    return dict(joints={n:float(joints[n]) for n in NAMES}, attached=attached,
        producer_stamps_ns={n:int(stamps[n]) for n in NAMES}, evaluated_ros_ns=now,
        contact_epoch=int(getattr(node,'_contact_epoch',0)),target_model=node._target_book_model,
        odom=None if odom is None else dict(odom))


def capture(node, **options):
    fault = node._payload_hazard_reason(max_age=.15)
    if fault is not None:
        raise RuntimeError('head_return_retention:' + str(fault))
    with node._lock:
        return snapshot_locked(node, **options)


def planar_radius(node, q, attached, *, right, head):
    """Original radius arithmetic, with both parked groups explicitly frozen."""
    transforms = node._collision_link_transforms(q, right_positions=right, head_positions=head)
    grasp = node.chain.forward(q)
    corners = attached @ grasp[:3,:3].T + grasp[:3,3]
    radius = float(np.max(np.linalg.norm(corners[:,:2],axis=1)))
    for mesh in node.carried_collision_meshes:
        transform = transforms[mesh.link]
        vertices = mesh.triangles.reshape(-1,3) @ transform[:3,:3].T + transform[:3,3]
        radius = max(radius,float(np.max(np.linalg.norm(vertices[:,:2],axis=1))))
    return radius


def admit_sweep(node, snapshot):
    joints=snapshot['joints'];q=np.asarray([joints[n] for n in IK_JOINTS]);right=np.asarray([joints[n] for n in RIGHT_ARM_JOINTS])
    start=np.asarray([joints[n] for n in HEAD]);goal=np.asarray(TARGET);attached=snapshot['attached']
    if (np.any(goal < node.head_chain.lower[1:]) or np.any(goal > node.head_chain.upper[1:])
            or np.any(start < node.head_chain.lower[1:]) or np.any(start > node.head_chain.upper[1:])):
        raise RuntimeError('head_return_head_limit')
    hand=node.chain.forward(q);corners=attached @ hand[:3,:3].T+hand[:3,3]
    context=dict(right_positions=right,head_positions=start)
    if (node._robot_self_collision(q,**context) is not None
            or node._carried_robot_collision(q,corners,**context) is not None):
        raise RuntimeError('head_return_current_carry_collision')
    count=node.carried_transition_samples
    if type(count) is not int or count < 3:
        raise RuntimeError('head_return_invalid_sample_policy')
    baseline=planar_radius(node,q,attached,right=right,head=start);maximum=baseline
    for fraction in np.linspace(0.,1.,count):
        head=start+(goal-start)*fraction
        if node._head_motion_collision(q,head,corners,right_positions=right) is not None:
            raise RuntimeError('head_return_head_sweep_collision')
        maximum=max(maximum,planar_radius(node,q,attached,right=right,head=head))
    if not math.isfinite(maximum) or maximum > baseline or maximum > node.carried_navigation_radius_limit:
        raise RuntimeError('head_return_navigation_envelope_expanded')
    return dict(samples=count,baseline_radius_m=baseline,maximum_radius_m=maximum,
        policy_radius_m=node.carried_navigation_radius_limit,
        scope='Original sampled head/body/nominal payload scope and unchanged planar envelope; no full-world or dynamic-retention proof')


def monitor(node):
    with node._lock:
        state=getattr(node,'_head_return_state',None)
        if state is None or state.get('fault'):
            return
        reference=state['reference'];phase=state['phase']
    try:
        capture(node,reference=reference,moving_head=phase!='completed',target=phase=='completed',stopped=phase=='sending')
    except RuntimeError as error:
        with node._lock:
            if getattr(node,'_head_return_state',None) is not state or state.get('fault'):
                return
            state['fault']=str(error)
            node._payload_hazard_latched='head_return_context:' + str(error)
            node._cancel.set()
        from .stock_gripper_close import enabled as stock_enabled
        if stock_enabled(node):
            node._hold_adaptive_gripper('head_return_context:' + str(error))
        node._publish_status('payload_hazard',reason='head_return_context:' + str(error),**state['identity'])


def active_check(node, state, *, target=False, stopped=False):
    with node._lock:
        if getattr(node,'_head_return_state',None) is not state or state.get('fault'):
            raise RuntimeError('head_return_owner_or_fault_changed')
    return capture(node,reference=state['reference'],moving_head=not target,target=target,
        stopped=stopped or state['phase']=='sending')


def run(node,payload):
    """One head action; keep its carry monitor until matching post-stop look_bin."""
    if not getattr(node,'head_return_overlap_enabled',False):
        raise RuntimeError('head_return_disabled')
    ident=identity(payload)
    first=capture(node,stopped=True)
    if first['target_model']!=ident['target_model']:
        raise RuntimeError('head_return_wrong_payload')
    with node._lock:
        if (getattr(node,'_head_return_state',None) is not None or node._goal_handles
                or node._pending_retained_acceptances):
            raise RuntimeError('head_return_existing_goal_or_owner')
    proof=admit_sweep(node,first)
    if not node.head_client.wait_for_server(timeout_sec=min(5.,node.timeout)):
        raise RuntimeError('head_return_server_unavailable')
    # This record denotes an actual new action, never a fabricated hold success.
    state=dict(identity=ident,reference=first,phase='sending',fault=None,proof=proof)
    first['base_must_match']=True
    token=object();future=None;handle=None;terminal=None;known_terminal=False
    try:
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        from rclpy.duration import Duration
        from action_msgs.msg import GoalStatus
        goal=FollowJointTrajectory.Goal();goal.trajectory.joint_names=list(HEAD)
        point=JointTrajectoryPoint();point.positions=list(TARGET);point.time_from_start=Duration(seconds=1.2).to_msg()
        goal.trajectory.points=[point]
        with node._adaptive_command_guard():
            capture(node,reference=first,stopped=True)
            with node._lock:
                snapshot_locked(node,reference=first,stopped=True)
                if node._goal_handles or node._pending_retained_acceptances:
                    raise RuntimeError('head_return_goal_race')
                node._last_completed_head_target=None
                node._head_return_state=state
                node._pending_retained_acceptances.add(token)
                future=node.head_client.send_goal_async(goal)
        deadline=time.monotonic()+min(5.,node.timeout)
        while not future.done():
            active_check(node,state)
            if time.monotonic()>=deadline:
                raise TimeoutError('head_return_acceptance_timeout')
            time.sleep(.02)
        handle=future.result()
        if handle is None or type(handle.accepted) is not bool:
            raise RuntimeError('head_return_invalid_acceptance')
        if not handle.accepted:
            with node._lock:node._pending_retained_acceptances.discard(token)
            known_terminal=True
            raise RuntimeError('head_return_goal_rejected')
        with node._lock:
            node._goal_handles.append(handle);node._pending_retained_acceptances.discard(token)
        terminal=handle.get_result_async()
        active_check(node,state)
        with node._lock:
            first['base_must_match']=False
            state['phase']='moving'
        node._publish_status('head_return_accepted',command=COMMAND,geometry=proof,**ident)
        deadline=time.monotonic()+min(node.timeout+4.8,20.)
        while not terminal.done():
            active_check(node,state)
            if time.monotonic()>=deadline:
                raise TimeoutError('head_return_motion_timeout')
            time.sleep(.02)
        wrapped=terminal.result()
        known_terminal=node._valid_retained_terminal_result(wrapped)
        if not known_terminal or wrapped.status!=GoalStatus.STATUS_SUCCEEDED or wrapped.result.error_code!=0:
            raise RuntimeError('head_return_trajectory_failed')
        with node._lock:
            completed_ns=int(node.get_clock().now().nanoseconds)
            node._last_completed_head_target=(TARGET,completed_ns)
        # Exact stationary endpoint uses the same strict measured-head criteria
        # as the existing completed hold; action success alone is insufficient.
        deadline=time.monotonic()+min(2.,node.timeout)
        while True:
            try:
                achieved=active_check(node,state,target=True)
                if all(achieved['producer_stamps_ns'][n]>completed_ns for n in HEAD):
                    break
            except RuntimeError:
                # Retention/parked faults remain fatal; only head convergence waits.
                active_check(node,state)
            if time.monotonic()>=deadline:
                raise TimeoutError('head_return_measured_head_timeout')
            time.sleep(.02)
        with node._lock:state['phase']='completed'
        node._publish_status('head_return_measured',command=COMMAND,producer_stamps_ns=achieved['producer_stamps_ns'],**ident)
        return True
    except Exception:
        node._cancel.set()
        if handle is not None and getattr(handle,'accepted',None) is True and not known_terminal:
            try:
                if terminal is None:terminal=handle.get_result_async()
                known_terminal=node._cancel_retained_goal_and_confirm(handle,terminal)
            except Exception:known_terminal=False
        elif future is not None and handle is None:
            try:future.add_done_callback(lambda f:node._cancel_late_retained_goal(f,token))
            except Exception:pass
        raise
    finally:
        if known_terminal:
            with node._lock:
                node._pending_retained_acceptances.discard(token)
                if handle in node._goal_handles:node._goal_handles.remove(handle)


def finish(node,payload):
    ident=identity(payload)
    with node._lock:
        state=getattr(node,'_head_return_state',None)
        if state is None or ident!=state['identity'] or state['phase']!='completed':
            raise RuntimeError('head_return_poststop_identity_or_completion')
        if node._goal_handles or node._pending_retained_acceptances:
            raise RuntimeError('head_return_poststop_goal_unresolved')
    active_check(node,state,target=True,stopped=True)
    # Preserve the ordinary measured completed-hold/fresh-start path. The
    # original entire sweep remains the checked envelope for a bounded fallback.
    with node._lock:state['phase']='poststop'
    if not node._move_head(*TARGET):return False
    active_check(node,state,target=True,stopped=True)
    with node._lock:
        if getattr(node,'_head_return_state',None) is not state or state.get('fault'):
            raise RuntimeError('head_return_poststop_context_race')
        node._head_return_state=None
    return True


def begin_mission(manager,goal):
    request=dict(trial_id=manager.trial_id,head_return_id=uuid.uuid4().hex,target_model=manager.target_book_model)
    identity(request)
    manager._head_return_request=dict(identity=request,goal=tuple(goal),accepted=False,succeeded=False,
        failure=None,nav_started=False,poststop=False)
    manager._perception_mode('idle')
    manager._manipulate(COMMAND,head_return_id=request['head_return_id'],target_model=request['target_model'])
    manager._set_state('RETURN_START')


def mission_status(manager,payload):
    if payload.get('command')!=COMMAND:
        return False
    state=getattr(manager,'_head_return_request',None)
    if state is None or any(payload.get(k)!=v for k,v in state['identity'].items()):
        manager._log('head_return_status_rejected',reason='identity_mismatch',payload=payload)
        return True
    event=payload.get('event')
    if event=='head_return_accepted':state['accepted']=True
    elif event=='succeeded':state['succeeded']=True
    elif event in ('failed','rejected','cancelled'):state['failure']=str(payload.get('reason',event))
    manager._log('head_return_status',payload=payload)
    return True


def tick_mission(manager):
    state=getattr(manager,'_head_return_request',None)
    if state is None:return False
    if state['failure'] is not None:
        manager._abort('head_return_failed:'+state['failure']);return True
    if manager._elapsed_state()>min(manager.manipulation_timeout,manager.navigation_timeout):
        manager._abort('head_return_timeout');return True
    if state['accepted'] and not state['nav_started']:
        manager._navigate(*state['goal'],'return_start_with_book');state['nav_started']=True
    if state['nav_started'] and manager._nav_failed():
        manager._abort('return_navigation_failed');return True
    if state['nav_started'] and manager._nav_reached() and state['succeeded']:
        state['poststop']=True
        manager._manipulate('look_bin',head_return_id=state['identity']['head_return_id'],target_model=state['identity']['target_model'])
        manager._set_state('HEAD_BIN')
    return True


def bin_epoch(manager,mode,fields):
    state=getattr(manager,'_head_return_request',None)
    if mode=='bin' and state is not None and state.get('poststop'):
        stamp=int(manager.get_clock().now().nanoseconds)
        fields['head_return_not_before_ns']=stamp
        manager.bin_invalidated_ns=max(getattr(manager,'bin_invalidated_ns',-1),stamp)


def camera_epoch(node,payload):
    value=payload.get('head_return_not_before_ns')
    if value is None:
        return
    now=int(node.get_clock().now().nanoseconds)
    if payload.get('event')!='bin' or type(value) is not int or value<=0 or not -.05e9<=now-value<=.35e9:
        raise RuntimeError('head_return_invalid_camera_epoch')
    node._head_return_camera_epoch_ns=value
    node.bin_history.clear()
    node.bin_tracker.reset()
    node.bin_rgb_frames.clear()
    node.bin_depth_frames.clear()
