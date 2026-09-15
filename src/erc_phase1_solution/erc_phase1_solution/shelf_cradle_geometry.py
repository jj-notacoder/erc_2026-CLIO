"""Pure mesh checks for a held book and the complete left gripper.

The production arm/payload planner must still run its own checks. This module
adds the nine gripper collision links omitted by that planner. Finger contact
with the target is intentional; palm penetration is rejected. All coordinates
are in base_footprint, and shelf_plane is the nearest shelf face, not book face.

The nominal URDF mimic linkage is a planning model, not evidence of retention
or actual finger deflection. Runtime joint/contact/target tracking gates remain
necessary. No ROS node, simulator query, or controller command is created here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import (
    PreparedTriangleMesh,
    URDFChain,
    oriented_box_intersects_triangles,
    triangle_meshes_intersect,
)
from .static_pair_separation import separated_on_axes
from .rigid_palm_preflight import (
    PALM_COLLISION_LINK,
    load_left_gripper_collision_model,
    world_gripper_surfaces,
)


class ShelfCradleGeometry:
    """Cache official meshes and finger FK; reuse across route sections."""

    def __init__(self, urdf: Path, resolve_package: Callable[[str], str | Path], *, immutable_local: bool = False):
        if type(immutable_local) is not bool:
            raise ValueError('immutable_local must be a Boolean')
        self.model = load_left_gripper_collision_model(urdf, resolve_package)
        self.chains = {
            mesh.link: URDFChain.from_urdf(urdf, 'base_footprint', mesh.link)
            for mesh in self.model.meshes
        }
        self.grasp_chain = URDFChain.from_urdf(
            urdf, 'base_footprint', 'gripper_left_grasping_link'
        )
        self.mimics = {
            joint.get('name'): joint.find('mimic')
            for joint in ET.parse(urdf).getroot().findall('joint')
            if joint.find('mimic') is not None
        }
        self.watertight = {
            link: all(mesh.watertight for mesh in self.model.meshes_for_link(link))
            for link in self.chains
        }
        self._local_cache: dict[tuple, dict[str, np.ndarray]] = {}
        self._local_mesh_cache = None
        if immutable_local:
            from .tool_local_meshes import ToolLocalMeshes
            self._local_mesh_cache = ToolLocalMeshes()

    def local_surfaces(
        self, aperture: float, finger_positions: Mapping[str, float] | None = None
    ) -> dict[str, np.ndarray]:
        """Resolve nominal mimic values, or use explicitly supplied joint values."""
        values = dict(finger_positions or {})
        values.setdefault('gripper_left_finger_joint', float(aperture))
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError('nonfinite gripper joint position')
        key = tuple(sorted(values.items()))
        registry = getattr(self, '_local_mesh_cache', None)
        if key in self._local_cache:
            cached = self._local_cache[key]
            return cached if registry is None else registry.materialize(cached)

        def value(name: str) -> float:
            if name in values:
                return values[name]
            mimic = self.mimics.get(name)
            if mimic is None:
                return 0.0
            return (
                value(mimic.get('joint')) * float(mimic.get('multiplier', '1'))
                + float(mimic.get('offset', '0'))
            )

        transforms = {
            link: chain.forward([value(name) for name in chain.active_names])
            for link, chain in self.chains.items()
        }
        grasp = self.grasp_chain.forward(
            [value(name) for name in self.grasp_chain.active_names]
        )
        surfaces = {
            link: (triangles - grasp[:3, 3]) @ grasp[:3, :3]
            for link, triangles in world_gripper_surfaces(self.model, transforms).items()
        }
        cached = surfaces if registry is None else registry.freeze(surfaces)
        self._local_cache[key] = cached
        return surfaces if registry is None else registry.materialize(cached)

    def model_local_surface(self, surface):
        """Opt in only the unchanged built-in producer and immutable buffer."""
        if (type(self) is not ShelfCradleGeometry
                or getattr(self.local_surfaces, '__func__', None) is not _NOMINAL_LOCAL_SURFACES):
            return None
        registry = getattr(self, '_local_mesh_cache', None)
        return None if registry is None else registry.model_for_surface(surface)


_NOMINAL_LOCAL_SURFACES = ShelfCradleGeometry.local_surfaces


def _bounds(surface: np.ndarray) -> np.ndarray:
    return np.asarray([surface.min(axis=(0, 1)), surface.max(axis=(0, 1))])


def check_cradle_tool_sweep(
    node,
    front: Sequence[float],
    grasp_solution: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
    shelf_plane: float | None,
    *,
    aperture: float | None = None,
    finger_positions: Mapping[str, float] | None = None,
    right_positions: Sequence[float] | None = None,
    head_positions: Sequence[float] | None = None,
) -> str | None:
    """Return a collision/failure reason, or None for a checked sampled sweep.

    Uses at least the production sample count and at most .02 rad per arm joint
    step. The shelf plane includes the existing carried_shelf_margin. A None
    plane permits extraction inside a shelf bay and therefore does NOT certify
    shelf clearance there. This check supplements the full arm/payload planner.
    """
    try:
        cancelled = getattr(node, '_cancel', None)
        if cancelled is not None and cancelled.is_set():
            return 'cradle_tool_cancelled'
        context = {}
        for name, supplied, size in (
            ('right_positions', right_positions, 7),
            ('head_positions', head_positions, 2),
        ):
            if supplied is not None:
                values = np.asarray(supplied, dtype=float)
                if values.shape != (size,) or not np.all(np.isfinite(values)):
                    return f'cradle_tool_{name}_invalid'
                context[name] = np.frombuffer(values.tobytes(), dtype=float)
        first, last = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        expected = (len(node.chain.active_names),)
        if first.shape != expected or last.shape != expected:
            return 'cradle_tool_joint_shape_invalid'
        if not np.all(np.isfinite([first, last])):
            return 'cradle_tool_joint_nonfinite'
        if (
            np.any(first < node.chain.lower) or np.any(first > node.chain.upper)
            or np.any(last < node.chain.lower) or np.any(last > node.chain.upper)
        ):
            return 'cradle_tool_joint_limit'
        point = np.asarray(front, dtype=float)
        dimensions = np.asarray(node.carried_book_dimensions, dtype=float)
        if (
            point.shape != (3,) or dimensions.shape != (3,)
            or not np.all(np.isfinite([point, dimensions]))
            or np.any(dimensions <= 0)
        ):
            return 'cradle_tool_book_geometry_invalid'
        if shelf_plane is not None and not np.isfinite(shelf_plane):
            return 'cradle_tool_shelf_plane_invalid'

        geometry = getattr(node, '_shelf_cradle_geometry', None)
        if geometry is None:
            from ament_index_python.packages import get_package_share_directory
            urdf = Path(get_package_share_directory('erc_description')) / 'urdf' / 'tiago_pro.urdf'
            geometry = ShelfCradleGeometry(urdf, get_package_share_directory, immutable_local=True)
            node._shelf_cradle_geometry = geometry
        # The adaptive lock setting may be a zero-valued lower bound, not the
        # held finger position. Nominal planning defaults to book thickness.
        position = float(aperture) if aperture is not None else float(dimensions[1])
        if not np.isfinite(position) or not 0 <= position <= .069:
            return 'cradle_tool_aperture_invalid'
        local = geometry.local_surfaces(position, finger_positions)
        grasp = node.chain.forward(grasp_solution)
        signs = np.asarray([
            (x, y, z) for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)
        ])
        centre = point + np.asarray([.5 * dimensions[0], 0., 0.])
        book = (centre + signs * (.5 * dimensions) - grasp[:3, 3]) @ grasp[:3, :3]
        # Book and tool remain rigid relative to the hand during this sweep.
        # Use actual book size: inflated grasp boxes overlap intended fingertips.
        if oriented_box_intersects_triangles(
            book, local[PALM_COLLISION_LINK],
            closed_surface=geometry.watertight[PALM_COLLISION_LINK],
        ):
            return 'cradle_book_palm_intersection'

        count = max(
            3, int(node.carried_transition_samples),
            int(math.ceil(float(np.max(np.abs(last - first))) / .02)) + 1,
        )
        if np.array_equal(first, last):
            count = 1  # A stationary pose has one unique geometric state.
        robot_watertight = node._watertight_collision_links()
        margin = float(node.carried_shelf_margin)
        for index, fraction in enumerate(np.linspace(0., 1., count)):
            cancelled = getattr(node, '_cancel', None)
            if cancelled is not None and cancelled.is_set():
                return 'cradle_tool_cancelled'
            joints = first + (last - first) * fraction
            # Share only the ordinary producer's exact current geometry.
            # Each sample owns a new snapshot; no verdict crosses a pose,
            # aperture, model or parked-arm/head context.
            from .sample_collision_snapshot import _capture_cradle_robot_snapshot
            snapshot = None
            if set(context) == {'right_positions', 'head_positions'}:
                snapshot = _capture_cradle_robot_snapshot(
                    node, joints, context['right_positions'], context['head_positions'])
            self_collision = node._robot_self_collision(
                joints, **context,
                **({'_scene_snapshot': snapshot} if snapshot is not None else {}))
            if self_collision is not None:
                return f'cradle_robot_self_intersection:{self_collision}:sample{index}'
            hand = node.chain.forward(joints)
            tool = {
                link: triangles @ hand[:3, :3].T + hand[:3, 3]
                for link, triangles in local.items()
            }
            shared = (None if snapshot is None else snapshot.resolved(
                node, joints, context['right_positions'], context['head_positions']))
            if shared is not None:
                robot, robot_bounds = shared
            else:
                robot = node._world_collision_surfaces(joints, **context)
                robot_bounds = {link: _bounds(surface) for link, surface in robot.items()}
            # Each exact sampled world surface can participate in multiple
            # mesh pairs. Keep preprocessing local to this sample; no pose
            # rounding, relative-frame shortcut or cross-aperture verdict reuse.
            prepared_tool, prepared_robot = {}, {}
            for robot_link, bounds in robot_bounds.items():
                if shelf_plane is not None and bounds[1, 0] > shelf_plane - margin:
                    return f'cradle_robot_shelf_clearance:{robot_link}:sample{index}'
                if robot_link.startswith('arm_left_') and bounds[0, 2] < .02:
                    return f'cradle_robot_floor_clearance:{robot_link}:sample{index}'
            for tool_link, triangles in tool.items():
                bounds = _bounds(triangles)
                if bounds[0, 2] < .02:
                    return f'cradle_tool_floor_clearance:{tool_link}:sample{index}'
                if shelf_plane is not None and bounds[1, 0] > shelf_plane - margin:
                    return f'cradle_tool_shelf_clearance:{tool_link}:sample{index}'
                for robot_link, surface in robot.items():
                    if tool_link == PALM_COLLISION_LINK and robot_link == 'arm_left_7_link':
                        continue  # The directly attached wrist/palm pair is adjacent.
                    other = robot_bounds[robot_link]
                    if np.any(bounds[1] < other[0]) or np.any(other[1] < bounds[0]):
                        continue
                    # A separating direction projects every current vertex.
                    # Uncertain and near-contact pairs retain the detailed
                    # surface and containment predicate below.
                    if separated_on_axes(triangles, surface, hand[:3, :3].T):
                        continue
                    if tool_link not in prepared_tool:
                        prepared_tool[tool_link] = PreparedTriangleMesh(triangles)
                    if robot_link not in prepared_robot:
                        prepared_robot[robot_link] = PreparedTriangleMesh(surface)
                    if triangle_meshes_intersect(
                        prepared_tool[tool_link], prepared_robot[robot_link],
                        first_watertight=geometry.watertight[tool_link],
                        second_watertight=robot_link in robot_watertight,
                    ):
                        return f'cradle_tool_robot_intersection:{tool_link}:{robot_link}:sample{index}'
        return None
    except Exception as exc:
        return f'cradle_tool_geometry_unavailable:{type(exc).__name__}:{exc}'


def check_cradle_tool_route(
    node,
    front: Sequence[float],
    grasp_solution: Sequence[float],
    start: Sequence[float],
    route: Sequence[Sequence[float]],
    shelf_plane: float | None,
    *,
    aperture: float | None = None,
    finger_positions: Mapping[str, float] | None = None,
) -> str | None:
    """Check every route section, merging only collinear joint-space segments.

    Compact goals contain many points on the same linear joint trajectory.
    Merging those sections preserves the path and the .02 rad sweep step while
    avoiding the minimum sample count being repeated at every intermediate goal.
    A change of direction or any noncollinear corner is retained.
    """
    try:
        points = [np.asarray(start, dtype=float)]
        for waypoint in route:
            current = np.asarray(waypoint, dtype=float)
            if current.shape != points[0].shape or not np.all(np.isfinite(current)):
                return 'cradle_tool_joint_shape_invalid'
            if np.allclose(current, points[-1], atol=1e-12, rtol=0):
                continue
            while len(points) >= 2:
                previous = points[-1] - points[-2]
                following = current - points[-1]
                denominator = float(np.dot(previous, previous))
                scale = float(np.dot(previous, following)) / denominator
                if scale < 0 or not np.allclose(
                    following, scale * previous, atol=1e-10, rtol=0,
                ):
                    break
                points.pop()
            points.append(current)
        if len(points) == 1:
            points.append(points[0])
        for index, (first, last) in enumerate(zip(points, points[1:])):
            reason = check_cradle_tool_sweep(
                node, front, grasp_solution, first, last, shelf_plane,
                aperture=aperture, finger_positions=finger_positions,
            )
            if reason:
                return f'leg{index}:{reason}'
        return None
    except Exception as exc:
        return f'cradle_tool_geometry_unavailable:{type(exc).__name__}:{exc}'


def check_gripper_opening(
    node,
    front: Sequence[float],
    grasp_solution: Sequence[float],
    joints: Sequence[float],
    shelf_plane: float | None,
    *,
    start_aperture: float = 0.,
    end_aperture: float = .069,
) -> str | None:
    """Check nominal empty-gripper opening at one arm pose in <=1mm steps."""
    if (
        not np.all(np.isfinite([start_aperture, end_aperture]))
        or not 0 <= start_aperture <= .069
        or not 0 <= end_aperture <= .069
    ):
        return 'cradle_tool_aperture_invalid'
    count = int(math.ceil(abs(end_aperture - start_aperture) / .001)) + 1
    for aperture in np.linspace(start_aperture, end_aperture, count):
        reason = check_cradle_tool_sweep(
            node, front, grasp_solution, joints, joints, shelf_plane,
            aperture=float(aperture),
        )
        if reason:
            return f'aperture{aperture:.6f}:{reason}'
    return None
