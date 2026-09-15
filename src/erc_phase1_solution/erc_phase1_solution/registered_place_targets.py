"""Finite camera-registered PLACE targets preserving the worst wall reserve.

These are Cartesian proposals, never admitted robot paths. Each proposal keeps
orientation and release/clearance height, and all eight padded book corners stay
inside the same shrunken cavity when projected onto its horizontal axes. Full
robot, payload, tool, support, scene and opening checks remain the caller's job.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from .bin_scene_admission import validate_bin_scene
from .book_centered_place import BookCenteredPlaceTarget, book_centered_place_target


@dataclass(frozen=True)
class RegisteredPlaceProposal:
    positions: tuple
    diagnostics: dict
    target: BookCenteredPlaceTarget


def _array(value, shape, name):
    result = np.asarray(value, dtype=float).copy()
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError('invalid registered target ' + name)
    return result


def _interpolate(first, last, step):
    # Match the ordinary Cartesian planner, including its supplied endpoint.
    count = max(1, int(math.ceil(float(np.linalg.norm(last-first))/step)))
    return [first+(last-first)*(index/count) for index in range(1, count+1)]


def registered_place_proposals(positions, rotation, attached_corners, bin_scene, *, step):
    """Center first, then increasing near-side shifts at 25 mm resolution.

    Shift is along the measured CAD long axis toward the robot, clipped before
    reducing the nominal minimum horizontal wall reserve. Thus a long cavity's
    spare depth can improve reach without spending the book's tightest margin.
    Canonical across/above/release poses are rebuilt at the original step, then
    each segment receives one midpoint. Every original segment endpoint remains.
    Only the admitted measured-positive caller uses these optional proposals.
    """
    registered = validate_bin_scene(bin_scene)
    raw = np.asarray(positions, dtype=float)
    if (raw.ndim != 2 or raw.shape[1:] != (3,) or not 3 <= len(raw) <= 64
            or not np.all(np.isfinite(raw))):
        raise ValueError('invalid registered target positions')
    points = raw.copy()
    orientation = _array(rotation, (3, 3), 'rotation')
    attached = _array(attached_corners, (8, 3), 'attachment')
    step = float(step)
    if not math.isfinite(step) or not 0.005 <= step <= 0.06:
        raise ValueError('invalid registered target Cartesian step')
    frame = np.asarray(registered['rotation'])
    origin = np.asarray(registered['origin'])
    floor = np.asarray(registered['floor_center'])
    center = orientation @ attached.mean(axis=0) + points[-1]
    height = float((center-floor) @ frame[:, 1])
    nominal = book_centered_place_target(attached, floor, height, bin_rotation=frame)
    if (not np.allclose(nominal.tool_rotation, orientation, rtol=0., atol=1e-8)
            or not np.allclose(nominal.tool_position, points[-1], rtol=0., atol=1e-8)):
        raise ValueError('registered target is not the original centered pose')
    above = points[-1].copy(); above[2] = points[0, 2]
    clearance = above.copy(); clearance[0] = points[0, 0]
    if above[2] <= points[-1, 2] or above[0] <= clearance[0]:
        raise ValueError('invalid registered target approach direction')
    original = [clearance, *_interpolate(clearance, above, step),
                *_interpolate(above, points[-1], step)]
    if (len(original) != len(points)
            or not np.allclose(original, points, rtol=0., atol=1e-8)):
        raise ValueError('unsupported registered target Cartesian shape')
    # Identical effective material margin to NominalBinObstacle's registered
    # branch; validate_bin_scene already enforces the official >=5 mm model.
    margin = max(.005, registered['modeled_margin_m']) + registered['registration_margin_m']
    cavity = np.asarray(registered['cavity_bounds'])
    inset = cavity + np.asarray([[margin]*3, [-margin]*3])
    local = (attached @ orientation.T + points[-1] - origin) @ frame
    axes = [0, 2]

    def reserve(corners):
        return min(float((corners[:, axes]-inset[0, axes]).min()),
                   float((inset[1, axes]-corners[:, axes]).min()))

    original_reserve = reserve(local)
    if original_reserve < 0.:
        raise ValueError('centered padded book misses the shrunken cavity')
    near_dot = float(center[:2] @ frame[:2, 2])
    if abs(near_dot) < 1e-6:
        raise ValueError('ambiguous registered target near-side direction')
    direction = -1. if near_dot > 0. else 1.
    near_slack = (float(local[:, 2].min()-inset[0, 2]) if direction < 0.
                  else float(inset[1, 2]-local[:, 2].max()))
    maximum = max(0., near_slack-original_reserve)
    if maximum > .25:
        raise ValueError('unsupported registered target shift extent')
    offsets = [0.]
    for index in range(1, int(math.ceil(maximum/.025))+1):
        offset = min(index*.025, maximum)
        if offset-offsets[-1] > 1e-12:
            offsets.append(offset)
    proposals = []
    for offset in offsets:
        displacement = frame[:, 2] * direction * offset
        shifted = local.copy(); shifted[:, 2] += direction*offset
        selected_reserve = reserve(shifted)
        if selected_reserve < original_reserve-1e-12:
            raise ValueError('registered target reduced the original wall reserve')
        release = points[-1] + displacement
        high = above + displacement
        first = high.copy(); first[0] = clearance[0]
        coarse = [first, *_interpolate(first, high, step), *_interpolate(high, release, step)]
        refined = [coarse[0].copy()]
        for left, right in zip(coarse, coarse[1:]):
            refined.extend([(left+right)/2., right.copy()])
        if len(refined) > 64:
            raise ValueError('registered target exceeds bounded Cartesian count')
        for point in refined:
            point.flags.writeable = False
        diagnostics = dict(policy='registered_near_side_preserve_worst_reserve_v1',
            offset_m=float(offset), displacement=displacement.tolist(),
            maximum_shift_m=maximum, original_minimum_wall_reserve_m=original_reserve,
            selected_minimum_wall_reserve_m=selected_reserve, effective_material_margin_m=margin,
            projected_cavity_axes=axes, projected_book_corners=shifted[:, axes].tolist(),
            shrunken_cavity_horizontal_bounds=inset[:, axes].tolist(),
            book_center=(center+displacement).tolist(), tool_position=release.tolist(),
            tool_rotation=orientation.tolist(), clearance_position=high.tolist(),
            original_cartesian_step_m=step, segment_subdivision=2,
            coarse_cartesian_waypoints=len(coarse), cartesian_waypoints=len(refined))
        proposals.append(RegisteredPlaceProposal(tuple(refined), diagnostics,
            replace(nominal, book_center=center+displacement, tool_position=release.copy())))
    return tuple(proposals)
