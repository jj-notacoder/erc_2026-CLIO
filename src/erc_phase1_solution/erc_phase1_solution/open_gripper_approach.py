"""Nominal open-tool collision checks against a resting, upright target book.

The caller supplies the fresh perception front and the solved clearance-to-grasp
joint path in base_footprint. The book remains fixed while all nine official
gripper meshes move at the commanded open position. No held-book inflation is
appropriate here. This sampled check supplements the arm/shelf/neighbor guards;
it does not certify actual passive-finger positions or physical retention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .kinematics import oriented_box_intersects_triangles
from .rigid_palm_preflight import LEFT_GRIPPER_COLLISION_LINKS, PALM_COLLISION_LINK
from .shelf_cradle_geometry import ShelfCradleGeometry


@dataclass(frozen=True)
class OpenGripperApproachResult:
    ok: bool
    reason: str
    samples_checked: int = 0
    collision_link: str | None = None
    leg_index: int | None = None
    sample_index: int | None = None
    min_palm_front_gap_m: float | None = None


def check_open_gripper_approach(
    node,
    front: Sequence[float],
    solutions: Sequence[Sequence[float]],
    *,
    torso_height: float | None = None,
) -> OpenGripperApproachResult:
    """Reject any nominal open-tool/book overlap before closure or motion.

    Every supplied leg includes both endpoints and at least 61 samples, with
    joint increments no larger than .02. The optional torso value also binds
    the path to the planned fixed torso height. The caller must fail on !ok.
    """
    checked = 0
    minimum_gap = math.inf

    def result(reason, *, link=None, leg=None, sample=None):
        return OpenGripperApproachResult(
            reason == 'open_approach_clear', reason, checked, link, leg, sample,
            minimum_gap if math.isfinite(minimum_gap) else None,
        )

    try:
        names = tuple(node.chain.active_names)
        lower = np.asarray(node.chain.lower, dtype=float)
        upper = np.asarray(node.chain.upper, dtype=float)
        expected = (len(names),)
        if (not names or len(set(names)) != len(names)
                or lower.shape != expected or upper.shape != expected
                or not np.all(np.isfinite([lower, upper]))
                or np.any(lower > upper)):
            return result('open_approach_joint_limits_invalid')
        route = np.asarray(solutions, dtype=float)
        if route.ndim != 2 or route.shape[0] < 2 or route.shape[1:] != expected:
            return result('open_approach_joint_shape_invalid')
        if not np.all(np.isfinite(route)):
            return result('open_approach_joint_nonfinite')
        if np.any(route < lower) or np.any(route > upper):
            return result('open_approach_joint_limit')
        if torso_height is not None:
            height = float(torso_height)
            if (not math.isfinite(height) or 'torso_lift_joint' not in names
                    or np.any(np.abs(route[:, names.index('torso_lift_joint')] - height) > 1e-9)):
                return result('open_approach_torso_mismatch')

        point = np.asarray(front, dtype=float)
        dimensions = np.asarray(node.carried_book_dimensions, dtype=float)
        if (point.shape != (3,) or dimensions.shape != (3,)
                or not np.all(np.isfinite([point, dimensions]))
                or np.any(dimensions <= 0)):
            return result('open_approach_book_geometry_invalid')
        aperture = float(node.gripper_open)
        if not math.isfinite(aperture) or not 0 <= aperture <= .069:
            return result('open_approach_aperture_invalid')

        geometry = getattr(node, '_shelf_cradle_geometry', None)
        if geometry is None:
            from ament_index_python.packages import get_package_share_directory
            urdf = Path(get_package_share_directory('erc_description')) / 'urdf/tiago_pro.urdf'
            geometry = ShelfCradleGeometry(urdf, get_package_share_directory, immutable_local=True)
            node._shelf_cradle_geometry = geometry
        local = geometry.local_surfaces(aperture)
        if set(local) != set(LEFT_GRIPPER_COLLISION_LINKS):
            return result('open_approach_tool_geometry_invalid')
        bounds = {}
        for link in LEFT_GRIPPER_COLLISION_LINKS:
            surface = np.asarray(local[link], dtype=float)
            if (surface.ndim != 3 or surface.shape[1:] != (3, 3)
                    or not len(surface) or not np.all(np.isfinite(surface))
                    or link not in geometry.watertight
                    or not isinstance(geometry.watertight[link], (bool, np.bool_))):
                return result('open_approach_tool_geometry_invalid')
            bounds[link] = (surface.min(axis=(0, 1)), surface.max(axis=(0, 1)))

        signs = np.asarray([(x, y, z) for x in (-1., 1.)
                            for y in (-1., 1.) for z in (-1., 1.)])
        centre = point + np.asarray([.5 * dimensions[0], 0., 0.])
        book = centre + signs * (.5 * dimensions)
        palm_vertices = np.asarray(local[PALM_COLLISION_LINK]).reshape(-1, 3)
        for leg, (first, last) in enumerate(zip(route[:-1], route[1:])):
            count = max(61, int(math.ceil(float(np.max(np.abs(last - first))) / .02)) + 1)
            for sample, fraction in enumerate(np.linspace(0., 1., count)):
                joints = first + fraction * (last - first)
                hand = np.asarray(node.chain.forward(joints), dtype=float)
                if (hand.shape != (4, 4) or not np.all(np.isfinite(hand))
                        or not np.allclose(hand[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
                        or not np.allclose(hand[:3, :3].T @ hand[:3, :3], np.eye(3), atol=1e-7, rtol=0)
                        or not np.isclose(np.linalg.det(hand[:3, :3]), 1., atol=1e-7, rtol=0)):
                    return result('open_approach_fk_invalid', leg=leg, sample=sample)
                checked += 1
                gap = float(point[0] - (np.max(palm_vertices @ hand[0, :3]) + hand[0, 3]))
                minimum_gap = min(minimum_gap, gap)
                local_book = (book - hand[:3, 3]) @ hand[:3, :3]
                book_lower, book_upper = local_book.min(axis=0), local_book.max(axis=0)
                for link in LEFT_GRIPPER_COLLISION_LINKS:
                    tool_lower, tool_upper = bounds[link]
                    # Exact disjoint AABBs can skip SAT, including containment:
                    # an enclosed box necessarily overlaps the whole-mesh AABB.
                    if (np.any(book_upper < tool_lower - 1e-9)
                            or np.any(tool_upper < book_lower - 1e-9)):
                        continue
                    if oriented_box_intersects_triangles(
                        local_book, local[link], closed_surface=geometry.watertight[link],
                    ):
                        return result('open_approach_book_intersection', link=link, leg=leg, sample=sample)
        return result('open_approach_clear')
    except Exception as exc:
        return result(f'open_approach_geometry_unavailable:{type(exc).__name__}:{exc}')
