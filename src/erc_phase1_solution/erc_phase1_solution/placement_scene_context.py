"""Bind a camera-registered scene to fresh stationary robot measurements."""
from __future__ import annotations

import math
import numpy as np


def measured_scene_context(node, reference=None):
    """Read fresh parked geometry/base; reject drift of a frozen PLACE frame.

    Left arm, torso and gripper are deliberately allowed to move. The base,
    right arm and head define the unchanged scene and parked collision model.
    """
    with node._lock:
        joints = dict(node.joints)
        stamps = dict(getattr(node, '_joint_stamps_ns', {}))
        odom = dict(getattr(node, '_staging_odom', None) or {})
    now = int(node.get_clock().now().nanoseconds)
    if not -.05e9 <= now-odom.get('stamp_ns', -10**18) <= .35e9:
        raise RuntimeError('placement_scene_stale_odometry')
    pose = np.asarray(odom.get('pose'), dtype=float)
    speeds = np.asarray([odom.get('linear_speed'), odom.get('angular_speed')], dtype=float)
    if (pose.shape != (3,) or not np.all(np.isfinite(pose))
            or not np.all(np.isfinite(speeds)) or np.any(speeds < 0)
            or speeds[0] > .005 or speeds[1] > .008):
        if getattr(node, 'place_transition_stop_enabled', False):
            from .place_transition_stop import remember_scene_failure
            remember_scene_failure(node, now, odom, pose, speeds)
        raise RuntimeError('placement_scene_base_not_stationary')
    parked = {}
    for name in (*node.right_chain.active_names, *node.head_chain.active_names):
        if name == 'torso_lift_joint':
            continue
        value = joints.get(name, math.nan)
        if not math.isfinite(value) or not -.05e9 <= now-stamps.get(name, -10**18) <= .35e9:
            raise RuntimeError('placement_scene_stale_joint:' + name)
        parked[name] = float(value)
    if reference is not None:
        previous = np.asarray(reference['base_pose'], dtype=float)
        angle = math.atan2(math.sin(pose[2]-previous[2]), math.cos(pose[2]-previous[2]))
        if np.linalg.norm(pose[:2]-previous[:2]) > .002 or abs(angle) > .005:
            raise RuntimeError('placement_scene_base_moved')
        if any(abs(value-reference['parked_joints'][name]) > .001 for name, value in parked.items()):
            raise RuntimeError('placement_scene_parked_geometry_moved')
    return dict(base_pose=pose.tolist(), parked_joints=parked, stamp_ns=now,
                odom_frame_id=odom.get('frame_id'),
                odom_child_frame_id=odom.get('child_frame_id'),
                odom_producer_stamp_ns=odom.get('producer_stamp_ns'))


def matching_table_scene(entry, stamp_ns, bin_point):
    """Only the table produced from this exact RGB-D bin observation qualifies."""
    if not isinstance(entry, dict) or entry.get('observation_stamp_ns') != stamp_ns:
        raise RuntimeError('placement_scene_waiting_for_matching_table')
    scene = entry.get('table_scene')
    if not isinstance(scene, dict) or scene.get('valid') is not True:
        reason = scene.get('reason', 'unavailable') if isinstance(scene, dict) else 'unavailable'
        raise RuntimeError('placement_scene_table_unregistered:' + str(reason))
    if scene.get('frame') != 'base_footprint':
        raise RuntimeError('placement_scene_table_frame_mismatch')
    point = np.asarray(entry.get('bin_floor_point_base'), dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)) or np.linalg.norm(point-bin_point) > .001:
        raise RuntimeError('placement_scene_bin_frame_mismatch')
    # Values came from a decoded JSON message and are owned by this callback.
    # Returning this entry does not mutate it when a later event replaces it.
    return scene
