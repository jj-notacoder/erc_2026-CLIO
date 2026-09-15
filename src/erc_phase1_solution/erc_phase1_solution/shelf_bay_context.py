"""Transfer visual shelf registration for the optional bounded lift diagnostic.

All positions are derived from marker RGB-D/TF and wheel odometry. The generous
uncertainty allowances below are engineering assumptions, not measured sensor
covariance or a certificate for the complete shelf collision mesh.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .lift_first_extraction import RelativeShelfBay


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'invalid shelf registration {name}')
    return result


def _stamp(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'invalid shelf registration {name}')
    return value


def make_bay_context(marker, outward_normal, physical_column,
                     observation_stamp_ns, dispatch_stamp_ns):
    """Serialize the successful visual registration with its original epoch."""
    result = dict(schema=1, source='onboard_marker_cloud', frame_id='odom',
                  marker_odom=_vector(marker, 3, 'marker').tolist(),
                  outward_normal_odom=_vector(outward_normal, 2, 'normal').tolist(),
                  physical_column=physical_column,
                  observation_stamp_ns=observation_stamp_ns,
                  dispatch_stamp_ns=dispatch_stamp_ns)
    _validate(result, dispatch_stamp_ns)
    return result


def _validate(context, now_ns):
    if (not isinstance(context, Mapping) or context.get('schema') != 1
            or context.get('source') != 'onboard_marker_cloud'
            or context.get('frame_id') != 'odom'):
        raise ValueError('missing or incompatible onboard shelf registration')
    marker = _vector(context.get('marker_odom'), 3, 'marker')
    normal = _vector(context.get('outward_normal_odom'), 2, 'normal')
    if abs(float(np.linalg.norm(normal)) - 1.) > 1e-4:
        raise ValueError('shelf registration normal must be unit length')
    column = context.get('physical_column')
    if isinstance(column, bool) or not isinstance(column, int) or not 1 <= column <= 5:
        raise ValueError('invalid visually registered physical shelf column')
    observed = _stamp(context.get('observation_stamp_ns'), 'observation epoch')
    dispatched = _stamp(context.get('dispatch_stamp_ns'), 'dispatch epoch')
    if not 0 <= now_ns - observed <= 120_000_000_000:
        raise ValueError('shelf visual registration is stale or from the future')
    if not observed <= dispatched <= now_ns or now_ns - dispatched > 2_000_000_000:
        raise ValueError('shelf registration command epoch is invalid or stale')
    return marker, normal, column, observed


def decode_bay_context(node, payload):
    """Transform a retained static registration into the measured current base."""
    if not isinstance(payload, Mapping):
        raise ValueError('lift-first pick requires onboard shelf registration')
    now = int(node.get_clock().now().nanoseconds)
    marker, outward, column, observed = _validate(payload.get('shelf_bay_context'), now)
    with node._lock:
        sample = getattr(node, '_staging_odom', None)
        odom = dict(sample) if isinstance(sample, Mapping) else None
    if odom is None or not -50_000_000 <= now - odom.get('stamp_ns', 0) <= 350_000_000:
        raise ValueError('lift-first shelf registration needs fresh odometry')
    pose = _vector(odom.get('pose'), 3, 'current base pose')
    speed = _vector([odom.get('linear_speed'), odom.get('angular_speed')], 2, 'base speed')
    if np.any(speed < 0) or speed[0] > .005 or speed[1] > .008:
        raise ValueError('lift-first shelf registration requires a stationary base')
    c, s = math.cos(pose[2]), math.sin(pose[2])
    rotation = np.array([[c, s], [-s, c]])
    base_marker = np.r_[rotation @ (marker[:2] - pose[:2]), marker[2]]
    inward = np.r_[rotation @ -outward, 0.]
    # A glyph ROI may be off the centre of its 300 mm card by up to 150 mm.
    # Reserve another 50 mm for registration/TF orientation and translation.
    # Roof is relative to initial upright support, not marker vertical position.
    return RelativeShelfBay(
        marker_center_base=base_marker, inward_axis_base=inward,
        physical_column=column, lateral_uncertainty_m=.200,
        roof_uncertainty_m=.010, assume_upright_supported=True,
        source=('onboard marker RGB-D/TF in odom; current odometry; '
                f'observation_age_seconds={(now-observed)/1e9:.6f}; '
                'engineering allowances: lateral 0.200m, relative roof 0.010m; '
                'initial upright supported book assumed; covariance unmeasured'),
    )
