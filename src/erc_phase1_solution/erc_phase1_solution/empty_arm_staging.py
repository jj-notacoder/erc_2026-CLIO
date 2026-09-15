"""Plan a checked empty-arm setup behind the final shelf approach position.

All calculations are pure. This module issues no arm or base commands. The
caller must establish the returned backoff, execute the checked arm path, then
guard a straight forward base advance with the reported robot/tool footprint.
The base must not rotate during that advance, and other obstacles still require
live sensing. Final shelf-plane clearance is the tightest bound along that
translation; body self-collision and floor clearance do not change.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator, Sequence

import numpy as np

from .shelf_cradle_geometry import (
    check_cradle_tool_route,
    check_cradle_tool_sweep,
)


# Each joint moves directly from its measured start to the selected IK target.
# These are orders, not stored seed-specific joint angles. Every returned path
# is checked for the current start, target, full robot state and shelf position.
DEFAULT_ARM_ORDERS = (
    (1, 3, 6, 2, 4, 5, 7),
    (1, 3, 4, 6, 2, 5, 7),
    (1, 3, 4, 5, 6, 2, 7),
    (1, 4, 6, 2, 3, 5, 7),
    (1, 6, 2, 3, 4, 5, 7),
    (6, 1, 2, 3, 4, 5, 7),
)


@dataclass(frozen=True)
class EmptyArmStagingPlan:
    """A sampled collision certificate and the straight-advance envelope."""

    waypoints: tuple[np.ndarray, ...]
    arm_order: tuple[int, ...]
    staging_backoff_m: float
    minimum_sampled_backoff_m: float
    final_shelf_plane_x: float
    setup_shelf_plane_x: float
    setup_maximum_robot_tool_x: float
    advance_robot_tool_bounds: np.ndarray
    advance_robot_tool_radius_m: float
    final_shelf_clearance_m: float
    setup_bound_samples: int
    aperture_m: float


def endpoint_order_route(
    active_names: Sequence[str],
    start: Sequence[float],
    goal: Sequence[float],
    arm_order: Sequence[int],
) -> tuple[np.ndarray, ...]:
    """Return the minimum number of monotone single-joint endpoint moves."""
    if sorted(arm_order) != list(range(1, 8)):
        raise ValueError('arm order must contain joints 1 through 7 exactly once')
    current, target = np.asarray(start, dtype=float).copy(), np.asarray(goal, dtype=float)
    expected = (len(active_names),)
    if current.shape != expected or target.shape != expected:
        raise ValueError('empty-arm start/goal joint shape is invalid')
    if not np.all(np.isfinite([current, target])):
        raise ValueError('empty-arm start/goal must be finite')
    indices = [active_names.index(f'arm_left_{joint}_joint') for joint in arm_order]
    other_indices = sorted(set(range(len(active_names))) - set(indices))
    if not np.allclose(current[other_indices], target[other_indices], atol=1e-10, rtol=0):
        raise ValueError('empty-arm setup requires fixed non-arm joints, including torso')
    route = []
    for index in indices:
        if abs(current[index] - target[index]) <= 1e-12:
            continue
        current = current.copy()
        current[index] = target[index]
        route.append(current)
    return tuple(route)


def _route_samples(node, start, route) -> Iterator[np.ndarray]:
    previous = np.asarray(start, dtype=float)
    for target in route:
        target = np.asarray(target, dtype=float)
        count = max(
            3, int(node.carried_transition_samples),
            math.ceil(float(np.max(np.abs(target - previous))) / .02) + 1,
        )
        if np.array_equal(previous, target):
            count = 1
        for fraction in np.linspace(0., 1., count):
            yield previous + (target - previous) * fraction
        previous = target
    if not route:
        yield previous


def robot_tool_envelope(node, joints, aperture: float = 0.) -> tuple[np.ndarray, float]:
    """Return exact mesh-vertex bounds and planar radius at one planned pose.

    A prior tool guard call must have initialized the official gripper model.
    The radius includes tool meshes, which the carried-radius helper omits.
    """
    model = getattr(node, '_shelf_cradle_geometry', None)
    if model is None:
        raise RuntimeError('full gripper geometry has not been initialized')
    transform = node.chain.forward(joints)
    surfaces = list(node._world_collision_surfaces(joints).values())
    surfaces.extend(
        triangles @ transform[:3, :3].T + transform[:3, 3]
        for triangles in model.local_surfaces(aperture).values()
    )
    if not surfaces:
        raise RuntimeError('robot/tool collision geometry is empty')
    lower = np.min([surface.min(axis=(0, 1)) for surface in surfaces], axis=0)
    upper = np.max([surface.max(axis=(0, 1)) for surface in surfaces], axis=0)
    radius = max(
        float(np.max(np.linalg.norm(surface[:, :, :2], axis=2)))
        for surface in surfaces
    )
    return np.asarray([lower, upper]), radius


def plan_empty_arm_staging(
    node,
    front: Sequence[float],
    grasp_solution: Sequence[float],
    start: Sequence[float],
    goal: Sequence[float],
    final_shelf_plane_x: float,
    *,
    maximum_backoff_m: float = .5,
    positioning_reserve_m: float = .01,
    backoff_resolution_m: float = .01,
    aperture: float = 0.,
    arm_orders: Sequence[Sequence[int]] = DEFAULT_ARM_ORDERS,
) -> EmptyArmStagingPlan:
    """Find a complete checked setup and final straight-advance certificate.

    The supplied front/grasp/goal are expressed in the FINAL base frame. The
    setup executes the same robot-relative joints from farther back. Only the
    shelf plane moves away during setup; no payload is held. The future grasp
    is also checked for nominal book/palm penetration by the shared guard.

    Coarse samples reject candidates cheaply; they never authorize execution.
    An accepted route always passes the full production sweep and hard limits.
    The extra reserve covers staging-position error; finite collision samples
    remain finite samples, not a proof over all continuous configurations.
    """
    scalars = [final_shelf_plane_x, maximum_backoff_m, positioning_reserve_m,
               backoff_resolution_m, aperture, node.carried_shelf_margin]
    if not np.all(np.isfinite(scalars)):
        raise ValueError('empty-arm staging parameters must be finite')
    if (
        maximum_backoff_m < 0 or positioning_reserve_m < 0
        or backoff_resolution_m <= 0 or not 0 <= aperture <= .069
        or node.carried_shelf_margin < 0
    ):
        raise ValueError('empty-arm staging distances/aperture are invalid')
    first = np.asarray(start, dtype=float)
    target = np.asarray(goal, dtype=float)
    # Reject an unsafe advance endpoint before spending time on setup paths.
    reason = check_cradle_tool_sweep(
        node, front, grasp_solution, target, target, final_shelf_plane_x,
        aperture=aperture,
    )
    if reason:
        raise RuntimeError(f'empty-arm advance endpoint rejected: {reason}')
    advance_bounds, advance_radius = robot_tool_envelope(node, target, aperture)
    rejected = []
    coarse_cache: dict[tuple[float, ...], str | None] = {}
    far_plane = final_shelf_plane_x + maximum_backoff_m
    for order in arm_orders:
        route = endpoint_order_route(node.chain.active_names, first, target, order)
        previous = first
        reason = None
        for waypoint in route:
            for fraction in np.linspace(0., 1., 9):
                sample = previous + (waypoint - previous) * fraction
                key = tuple(float(value) for value in sample)
                if key not in coarse_cache:
                    coarse_cache[key] = check_cradle_tool_sweep(
                        node, front, grasp_solution, sample, sample, far_plane,
                        aperture=aperture,
                    )
                reason = coarse_cache[key]
                if reason:
                    break
            if reason:
                break
            previous = waypoint
        if reason:
            rejected.append(f'{tuple(order)}: {reason}')
            continue
        maximum_x = -math.inf
        sample_count = 0
        for sample in _route_samples(node, first, route):
            bounds, _ = robot_tool_envelope(node, sample, aperture)
            maximum_x = max(maximum_x, float(bounds[1, 0]))
            sample_count += 1
        minimum_backoff = max(
            0., maximum_x + node.carried_shelf_margin - final_shelf_plane_x,
        )
        backoff = 0. if minimum_backoff == 0. else (
            math.ceil(
                (minimum_backoff + positioning_reserve_m - 1e-12)
                / backoff_resolution_m
            ) * backoff_resolution_m
        )
        if backoff > maximum_backoff_m + 1e-12:
            rejected.append(f'{tuple(order)}: requires {backoff:.4f}m backoff')
            continue
        reason = check_cradle_tool_route(
            node, front, grasp_solution, first, route,
            final_shelf_plane_x + backoff, aperture=aperture,
        )
        if reason:
            rejected.append(f'{tuple(order)}: {reason}')
            continue
        return EmptyArmStagingPlan(
            waypoints=tuple(waypoint.copy() for waypoint in route),
            arm_order=tuple(order), staging_backoff_m=float(backoff),
            minimum_sampled_backoff_m=float(minimum_backoff),
            final_shelf_plane_x=float(final_shelf_plane_x),
            setup_shelf_plane_x=float(final_shelf_plane_x + backoff),
            setup_maximum_robot_tool_x=float(maximum_x),
            advance_robot_tool_bounds=advance_bounds,
            advance_robot_tool_radius_m=advance_radius,
            final_shelf_clearance_m=float(final_shelf_plane_x - advance_bounds[1, 0]),
            setup_bound_samples=sample_count, aperture_m=float(aperture),
        )
    raise RuntimeError('no checked empty-arm staging route; ' + '; '.join(rejected))
