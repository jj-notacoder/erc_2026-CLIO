"""Pure admission for a placement-only, depth-epoch registered stock bin.

No ROS, mesh loading, interpolation, or control. Navigation's legacy point is
matched unchanged; the caller selects the returned bin floor center for PLACE.
Bounds are engineering allowances, not calibrated uncertainty certificates.
"""
from __future__ import annotations

import copy
import math
import numbers

import numpy as np

from .placement_scene_context import matching_table_scene

BIN_MODEL = 'stock_bin_visible_inner_planes_v1'
BIN_MESH_SHA256 = '7c036fe096a50eadc8c32f30837964c3bbbb012cf7a45b874eb7a6f96973aa2a'
OUTER_BOUNDS = ((-.155, -.105, -.305), (.155, .105, .255))
CAVITY_BOUNDS = ((-.145, -.095, -.245), (.145, .105, .245))
FLOOR_POINT_MODEL = (0., -.095, 0.)
CAD_RADIUS = math.sqrt(.155**2 + .105**2 + .305**2)


def _reject(reason):
    raise RuntimeError('placement_scene_bin_' + reason)


def _vector(value, shape, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        _reject('invalid_' + label)
    if result.shape != shape or not np.all(np.isfinite(result)):
        _reject('invalid_' + label)
    return result


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        _reject('invalid_' + label)
    return float(value)


def _stamp(value, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        _reject('invalid_' + label)
    return int(value)


def validate_bin_scene(scene):
    """Validate the supported CAD/fit contract and return independently owned data."""
    if not isinstance(scene, dict) or scene.get('valid') is not True:
        _reject('unregistered')
    result = copy.deepcopy(scene)
    if (result.get('model') != BIN_MODEL or result.get('frame') != 'base_footprint'
            or result.get('mesh_sha256') != BIN_MESH_SHA256):
        _reject('model_or_frame_mismatch')
    origin = _vector(result.get('origin'), (3,), 'origin')
    rotation = _vector(result.get('rotation'), (3, 3), 'rotation')
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=1e-7)
            or rotation[2, 1] < math.cos(math.radians(5.))):
        _reject('nonrigid_or_tilted_rotation')
    center = _vector(result.get('floor_center'), (3,), 'floor_center')
    floor = _vector(result.get('floor_point_model'), (3,), 'floor_model')
    outer = _vector(result.get('outer_bounds'), (2, 3), 'outer_bounds')
    cavity = _vector(result.get('cavity_bounds'), (2, 3), 'cavity_bounds')
    if (not np.array_equal(floor, FLOOR_POINT_MODEL)
            or not np.array_equal(outer, OUTER_BOUNDS)
            or not np.array_equal(cavity, CAVITY_BOUNDS)):
        _reject('cad_bounds_mismatch')
    if np.linalg.norm(center - (origin + rotation @ floor)) > 1e-8:
        _reject('floor_origin_mismatch')
    quality = result.get('quality')
    if not isinstance(quality, dict) or quality.get('floor_rear_registration_observed') is not True:
        _reject('unidentified_cad_orientation')
    candidates = quality.get('accepted_candidates')
    if isinstance(candidates, bool) or not isinstance(candidates, numbers.Integral) or candidates <= 0:
        _reject('unidentified_cad_orientation')
    # This model rejects incompatible candidate poses before returning valid.
    # Multiple compatible feature associations are allowed, not averaged here.
    if quality.get('ambiguous_cad_registration', False) is not False:
        _reject('ambiguous_cad_registration')
    coverage = _number(quality.get('finite_cad_coverage'), 'cad_coverage')
    residual = _number(quality.get('finite_surface_residual_p95_m'), 'fit_residual')
    if not .95 <= coverage <= 1. or not 0. <= residual <= .003:
        _reject('unsupported_fit_quality')
    modeled = _number(result.get('modeled_margin_m'), 'modeled_margin')
    registration = _number(result.get('registration_margin_m'), 'registration_margin')
    translation = _number(result.get('translation_error_m'), 'translation_error')
    angle = _number(result.get('rotation_error_rad'), 'rotation_error')
    if (modeled < .005 or not .005 + residual <= translation <= .008 or angle != .01
            or registration < translation + 2. * CAD_RADIUS * math.sin(angle / 2.)
            or not .005 <= modeled + registration <= .12):
        _reject('insufficient_or_excessive_margin')
    return result


def match_registered_scenes(entry, stamp_ns, legacy_bin_point, current_context):
    """Bind both scenes to one fresh depth epoch and stationary current base.

    current_context comes from measured_scene_context, whose parked-joint,
    current base-speed and freshness checks remain the caller's responsibility.
    No new semantics apply to callers that keep matching_table_scene alone.
    """
    stamp = _stamp(stamp_ns, 'observation_stamp')
    point = _vector(legacy_bin_point, (3,), 'legacy_point')
    if not isinstance(entry, dict) or not isinstance(current_context, dict):
        _reject('missing_epoch_binding')
    bundle = copy.deepcopy(entry)
    context = copy.deepcopy(current_context)
    try:
        table = matching_table_scene(bundle, stamp, point)
    except (TypeError, ValueError, OverflowError):
        _reject('invalid_matching_table')
    if _stamp(bundle.get('observation_stamp_ns'), 'entry_stamp') != stamp:
        _reject('entry_stamp_mismatch')
    scene = validate_bin_scene(bundle.get('bin_scene'))
    reference = bundle.get('scene_base_reference')
    if not isinstance(reference, dict):
        _reject('missing_epoch_binding')
    if (reference.get('frame_id') != 'odom' or reference.get('child_frame_id') != 'base_footprint'
            or context.get('odom_frame_id') != 'odom'
            or context.get('odom_child_frame_id') != 'base_footprint'):
        _reject('odometry_frame_mismatch')
    requested = _stamp(reference.get('requested_stamp_ns'), 'requested_stamp')
    producer = _stamp(reference.get('producer_stamp_ns'), 'transform_stamp')
    if requested != stamp or producer != stamp:
        _reject('transform_epoch_mismatch')
    now = _stamp(context.get('stamp_ns'), 'current_context_stamp')
    odom_stamp = _stamp(context.get('odom_producer_stamp_ns'), 'odometry_stamp')
    if not 0 <= now - stamp <= 500_000_000:
        _reject('observation_not_fresh')
    if not 0 <= now - odom_stamp <= 350_000_000:
        _reject('odometry_not_fresh')
    previous = _vector(reference.get('pose'), (3,), 'image_base_pose')
    current = _vector(context.get('base_pose'), (3,), 'current_base_pose')
    angle = math.atan2(math.sin(current[2] - previous[2]), math.cos(current[2] - previous[2]))
    if np.linalg.norm(current[:2] - previous[:2]) > .002 or abs(angle) > .005:
        _reject('base_moved_since_image')
    return copy.deepcopy(table), scene
