"""Measured, empty-handed setup for the opt-in shelf-side support experiment."""

from __future__ import annotations

import math
import time

import numpy as np

from .motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS, shelf_pinch_orientations
from .empty_arm_staging import plan_empty_arm_staging, robot_tool_envelope
from .shelf_cradle_geometry import (
    check_cradle_tool_sweep, check_gripper_opening,
)


def angle_error(first, second):
    return math.atan2(math.sin(first - second), math.cos(first - second))


def point_in_pose(point, pose):
    """Transform an odom point into a planar base frame, preserving height."""
    dx, dy = np.asarray(point, dtype=float)[:2] - np.asarray(pose, dtype=float)[:2]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, float(point[2])])


def validate_preparation_request(payload, now_ns, measured_pose, max_age=2.0):
    plan_id = payload.get('plan_id')
    if not isinstance(plan_id, str) or not 1 <= len(plan_id) <= 128:
        raise ValueError('empty-arm preparation requires a plan_id')
    arrays = []
    for key in ('book_odom', 'staging_pose', 'final_goal'):
        value = np.asarray(payload.get(key), dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f'invalid empty-arm {key}')
        arrays.append(value)
    book, start, goal = arrays
    age = (now_ns - int(payload.get('book_stamp_ns', 0))) / 1e9
    if not -0.05 <= age <= max_age:
        raise ValueError('empty-arm book observation is stale')
    measured = np.asarray(measured_pose, dtype=float)
    if (measured.shape != (3,) or not np.all(np.isfinite(measured))
            or np.linalg.norm(measured[:2] - start[:2]) > .012
            or abs(angle_error(measured[2], start[2])) > .008):
        raise ValueError('empty-arm staging pose changed')
    direction = np.array([math.cos(start[2]), math.sin(start[2])])
    delta = goal[:2] - start[:2]
    advance = float(delta @ direction)
    lateral = float(delta @ np.array([-direction[1], direction[0]]))
    if (not .04 <= advance <= .35 or abs(lateral) > .005
            or abs(angle_error(goal[2], start[2])) > .005):
        raise ValueError('empty-arm final goal must be a short straight advance')
    front = point_in_pose(book, goal)
    if not (.70 <= front[0] <= .90 and abs(front[1]) <= .15
            and 1.42 <= front[2] <= 1.85):
        raise ValueError('empty-arm future book point is outside the top-row workspace')
    return plan_id, book, start, goal, front, advance


def snapshot(node):
    with node._lock:
        return (dict(node.joints), dict(getattr(node, '_joint_stamps_ns', {})),
                getattr(node, '_staging_odom', None),
                getattr(node, '_empty_hand_contact', None))


def check_empty_stationary(node, expected_pose=None):
    """Require recent independent measurements; no pose fallback is allowed."""
    if (getattr(node, '_held_book_corners', None) is not None
            or getattr(node, '_transport_lock_engaged', False)):
        raise RuntimeError('empty-arm preparation requested while a payload may be held')
    if getattr(node, '_empty_arm_contact_latched', False):
        raise RuntimeError('external hand contact occurred during empty-arm preparation')
    joints, stamps, odom, contact = snapshot(node)
    now = node.get_clock().now().nanoseconds
    names = (*IK_JOINTS, *RIGHT_ARM_JOINTS, 'head_1_joint', 'head_2_joint',
             'gripper_left_finger_joint')
    for name in names:
        if (name not in joints or not math.isfinite(joints[name])
                or not -.05e9 <= now - stamps.get(name, 0) <= .35e9):
            raise RuntimeError(f'empty-arm joint measurement stale: {name}')
    if odom is None or not -.05e9 <= now - odom['stamp_ns'] <= .35e9:
        raise RuntimeError('empty-arm odometry is stale')
    if not np.all(np.isfinite([*odom['pose'], odom['linear_speed'], odom['angular_speed']])):
        raise RuntimeError('empty-arm odometry is invalid')
    if odom['linear_speed'] > .005 or odom['angular_speed'] > .008:
        raise RuntimeError('empty-arm base is moving')
    if expected_pose is not None:
        pose = odom['pose']
        if (np.linalg.norm(np.asarray(pose[:2]) - expected_pose[:2]) > .012
                or abs(angle_error(pose[2], expected_pose[2])) > .008):
            raise RuntimeError('empty-arm base moved during preparation')
    # Official per-link contact sensors are event-only: no empty heartbeat is
    # published. Publisher presence establishes availability, not emptiness.
    # The subsequent measured closure (<4 mm, against a 30 mm book) supplies the
    # empty-hand evidence; every external contact during staging is latched.
    if node.count_publishers('/contacts') <= 0:
        raise RuntimeError('empty-arm contact publisher is unavailable')
    positive_stamp = getattr(node, '_last_external_hand_contact_ns', None)
    if positive_stamp is None and contact is not None and contact[1]:
        positive_stamp = contact[0]
    if positive_stamp is not None and -.05e9 <= now - positive_stamp <= 1.e9:
        raise RuntimeError('empty-arm hand has external contact')
    return joints, odom


def wait_stable(node, expected_pose=None, timeout=10.0):
    deadline = time.monotonic() + timeout
    baseline, since = None, None
    last_error = 'empty-arm measurements did not settle'
    while not node._cancel.is_set() and time.monotonic() < deadline:
        try:
            joints, odom = check_empty_stationary(node, expected_pose)
            now = node.get_clock().now().nanoseconds
            names = (*IK_JOINTS, *RIGHT_ARM_JOINTS, 'head_1_joint', 'head_2_joint',
                     'gripper_left_finger_joint')
            values = np.array([joints[name] for name in names])
            limits = np.array([.0005 if name == 'torso_lift_joint' else .002
                               for name in names])
            if baseline is None or np.any(np.abs(values - baseline) > limits):
                baseline, since = values, now
            elif now - since >= .35e9:
                return joints, odom
        except RuntimeError as exc:
            baseline, since = None, None
            last_error = str(exc)
        time.sleep(.025)
    raise RuntimeError(last_error)


def prepare_pick_approach(node, payload):
    if not getattr(node, 'shelf_side_cradle_enabled', False):
        raise RuntimeError('empty-arm staging requires the experimental cradle profile')
    if getattr(node, '_empty_arm_staged', False):
        raise RuntimeError('empty-arm setup already active; hold for recovery')
    joints, odom = wait_stable(node)
    now = node.get_clock().now().nanoseconds
    plan_id, book, start_pose, goal, front, advance = validate_preparation_request(
        payload, now, odom['pose'], node.perception_max_age,
    )
    # Solve the future Cartesian approach, then replace the old folded-arm
    # transition with a complete, independently checked empty-hand route.
    grasp = front + np.array([node.top_row_grasp_depth_offset,
                              node.top_row_grasp_lateral_offset,
                              node.top_row_grasp_vertical_offset])
    pregrasp = grasp.copy()
    pregrasp[0] -= node.pregrasp_offset
    clearance = pregrasp.copy()
    clearance[0] = node.top_row_cartesian_clearance
    clearance[2] += node.top_row_loaded_clearance_lift
    positions = [clearance]
    positions.extend(node._interpolate_positions(clearance, pregrasp, node.cartesian_step))
    positions.extend(node._interpolate_positions(pregrasp, grasp, node.cartesian_step))
    solutions, _, _, _ = node._solve_cartesian_path(
        positions, shelf_pinch_orientations(float(grasp[2])),
        node.pick_torso_height, endpoint_first=True, skip_setup_transition=True,
    )
    measured = node._measured_left_solution()
    elevated = measured.copy()
    elevated[0] = node.pick_torso_height
    final_plane = float(front[0]) - .065
    available = advance - .015
    plan = plan_empty_arm_staging(
        node, front, solutions[-1], elevated, solutions[0], final_plane,
        maximum_backoff_m=available,
    )
    # Use a slightly closer conservative shelf plane for setup to cover the
    # accepted staging position/heading uncertainty. Torso and fingers are
    # separate physical motions and receive their own complete sweeps.
    setup_plane = final_plane + available
    aperture = joints['gripper_left_finger_joint']
    reason = check_gripper_opening(
        node, front, solutions[-1], measured, setup_plane,
        start_aperture=aperture, end_aperture=0.,
    )
    if reason:
        raise RuntimeError(f'empty-arm finger closure rejected: {reason}')
    reason = check_cradle_tool_sweep(
        node, front, solutions[-1], measured, elevated, setup_plane, aperture=0.,
    )
    if reason:
        raise RuntimeError(f'empty-arm torso setup rejected: {reason}')
    # Planning can be slow. Measurements must still match the checked state.
    current, _ = wait_stable(node, start_pose)
    current_q = np.array([current[name] for name in IK_JOINTS])
    if np.max(np.abs(current_q - measured)) > .003:
        raise RuntimeError('empty-arm joints changed during planning')
    context_names = (*RIGHT_ARM_JOINTS, 'head_1_joint', 'head_2_joint',
                     'gripper_left_finger_joint')
    if any(abs(current[name] - joints[name]) > .003 for name in context_names):
        raise RuntimeError('empty-arm collision context changed during planning')
    node._publish_status('empty_arm_plan_ready', command='prepare_pick_approach',
                         plan_id=plan_id, arm_order=list(plan.arm_order),
                         minimum_backoff=plan.staging_backoff_m,
                         available_backoff=available, final_front=front.tolist())
    # From the first command onward, an error holds the current posture. A
    # generic HOME transition at the close shelf position is not certified.
    node._empty_arm_staged = True
    node._empty_arm_contact_latched = False
    node._empty_arm_motion_active = True
    node._empty_arm_contact_guard = True
    if not node._command_gripper(0.):
        raise RuntimeError('empty-arm finger closure failed')
    current, _ = wait_stable(node, start_pose)
    if current['gripper_left_finger_joint'] > .004:
        raise RuntimeError('empty-arm gripper did not close empty')
    if not node._move_torso(node.pick_torso_height, 2.5):
        raise RuntimeError('empty-arm torso setup failed')
    current, _ = wait_stable(node, start_pose)
    if abs(current['torso_lift_joint'] - node.pick_torso_height) > .002:
        raise RuntimeError('empty-arm torso did not track the plan')
    for q in plan.waypoints:
        check_empty_stationary(node, start_pose)
        if not node._move_arm_solution(q, 2.5):
            raise RuntimeError('empty-arm setup motion failed')
        current, _ = wait_stable(node, start_pose)
        actual_leg = np.array([current[name] for name in IK_JOINTS])
        if (abs(actual_leg[0] - q[0]) > .002
                or np.max(np.abs(actual_leg[1:] - q[1:])) > .008):
            raise RuntimeError('empty-arm waypoint did not track the plan')
        if any(abs(current[name] - joints[name]) > .003
               for name in (*RIGHT_ARM_JOINTS, 'head_1_joint', 'head_2_joint')):
            raise RuntimeError('empty-arm collision context changed during execution')
    current, _ = wait_stable(node, start_pose)
    actual = np.array([current[name] for name in IK_JOINTS])
    if np.max(np.abs(actual - solutions[0])) > .008:
        raise RuntimeError('empty-arm endpoint did not track the plan')
    reason = check_cradle_tool_sweep(
        node, front, solutions[-1], actual, actual, final_plane - .015,
        aperture=current['gripper_left_finger_joint'],
    )
    if reason:
        raise RuntimeError(f'empty-arm measured advance posture rejected: {reason}')
    measured_bounds, measured_radius = robot_tool_envelope(
        node, actual, current['gripper_left_finger_joint'],
    )
    current, _ = check_empty_stationary(node, start_pose)
    prepared_names = (*IK_JOINTS, *RIGHT_ARM_JOINTS, 'head_1_joint', 'head_2_joint',
                      'gripper_left_finger_joint')
    certificate = {
        'plan_id': plan_id, 'prepared_stamp_ns': node.get_clock().now().nanoseconds,
        'final_goal': goal.tolist(), 'staging_pose': start_pose.tolist(),
        'book_odom': book.tolist(), 'predicted_front': front.tolist(),
        'prepared_joints': {name: float(current[name]) for name in prepared_names},
        'advance_bounds': measured_bounds.tolist(),
        'advance_radius': measured_radius,
    }
    node._empty_arm_preparation = certificate
    return True
