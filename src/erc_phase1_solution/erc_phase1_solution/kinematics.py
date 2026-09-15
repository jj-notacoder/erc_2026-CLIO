"""Small URDF kinematics and damped-least-squares IK solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _numbers(value: Optional[str], default: Sequence[float]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=float)
    return np.asarray([float(part) for part in value.split()], dtype=float)


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation, dtype=float)
    return result


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis_arr)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis_arr / norm
    c, s, one = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.asarray(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True)
class CollisionMesh:
    """Triangle collision surface expressed in a URDF link frame."""

    link: str
    triangles: np.ndarray
    bounds: np.ndarray
    watertight: bool
    local_mesh: object = None


class URDFChain:
    """Kinematic chain traced from a URDF base link to an end-effector link."""

    def __init__(self, joints: Sequence[Joint], active_names: Sequence[str]):
        self.joints = list(joints)
        self.active_names = list(active_names)
        self._active_index = {name: index for index, name in enumerate(self.active_names)}
        by_name = {joint.name: joint for joint in self.joints}
        self.lower = np.asarray([by_name[name].lower for name in self.active_names])
        self.upper = np.asarray([by_name[name].upper for name in self.active_names])

    @classmethod
    def from_urdf(
        cls,
        path: str | Path,
        base_link: str,
        tip_link: str,
        active_names: Optional[Sequence[str]] = None,
    ) -> 'URDFChain':
        root = ET.parse(str(path)).getroot()
        child_map: Dict[str, Joint] = {}
        for element in root.findall('joint'):
            kind = element.get('type', 'fixed')
            parent_element = element.find('parent')
            child_element = element.find('child')
            if parent_element is None or child_element is None:
                continue
            parent = parent_element.get('link', '')
            child = child_element.get('link', '')
            origin_element = element.find('origin')
            xyz = _numbers(
                origin_element.get('xyz') if origin_element is not None else None,
                (0.0, 0.0, 0.0),
            )
            rpy = _numbers(
                origin_element.get('rpy') if origin_element is not None else None,
                (0.0, 0.0, 0.0),
            )
            axis_element = element.find('axis')
            axis = _numbers(
                axis_element.get('xyz') if axis_element is not None else None,
                (1.0, 0.0, 0.0),
            )
            limit = element.find('limit')
            if kind == 'continuous':
                lower, upper = -math.pi, math.pi
            elif kind in ('revolute', 'prismatic') and limit is not None:
                lower = float(limit.get('lower', '-3.141592653589793'))
                upper = float(limit.get('upper', '3.141592653589793'))
            else:
                lower = upper = 0.0
            # Keep the commanded URDF hard range.  The current live-proven
            # top-row branch has no continuous solution inside the legacy
            # safety_controller soft window; solve() instead reserves an
            # explicit interior margin from each hard stop.
            child_map[child] = Joint(
                name=element.get('name', child),
                parent=parent,
                child=child,
                kind=kind,
                origin=_transform(_rpy_matrix(rpy), xyz),
                axis=axis,
                lower=lower,
                upper=upper,
            )

        chain: List[Joint] = []
        cursor = tip_link
        visited = set()
        while cursor != base_link:
            if cursor in visited or cursor not in child_map:
                raise ValueError(f'No URDF chain from {base_link!r} to {tip_link!r}')
            visited.add(cursor)
            joint = child_map[cursor]
            chain.append(joint)
            cursor = joint.parent
        chain.reverse()

        movable = [j.name for j in chain if j.kind in ('revolute', 'continuous', 'prismatic')]
        selected = list(active_names) if active_names is not None else movable
        missing = set(selected) - set(movable)
        if missing:
            raise ValueError(f'Active joints not present in chain: {sorted(missing)}')
        return cls(chain, selected)

    def forward(self, positions: Sequence[float]) -> np.ndarray:
        q = np.asarray(positions, dtype=float)
        if q.shape != (len(self.active_names),):
            raise ValueError(f'Expected {len(self.active_names)} joint values, got {q.shape}')
        result = np.eye(4, dtype=float)
        for joint in self.joints:
            result = result @ joint.origin
            index = self._active_index.get(joint.name)
            value = 0.0 if index is None else float(q[index])
            if joint.kind in ('revolute', 'continuous'):
                result = result @ _transform(_axis_rotation(joint.axis, value), (0, 0, 0))
            elif joint.kind == 'prismatic':
                result = result @ _transform(np.eye(3), joint.axis * value)
        return result

    def link_transforms(self, positions: Sequence[float]) -> Dict[str, np.ndarray]:
        """Return base-frame transforms for every child link in the chain."""
        q = np.asarray(positions, dtype=float)
        if q.shape != (len(self.active_names),):
            raise ValueError(f'Expected {len(self.active_names)} joint values, got {q.shape}')
        result = np.eye(4, dtype=float)
        transforms: Dict[str, np.ndarray] = {}
        for joint in self.joints:
            result = result @ joint.origin
            index = self._active_index.get(joint.name)
            value = 0.0 if index is None else float(q[index])
            if joint.kind in ('revolute', 'continuous'):
                result = result @ _transform(_axis_rotation(joint.axis, value), (0, 0, 0))
            elif joint.kind == 'prismatic':
                result = result @ _transform(np.eye(3), joint.axis * value)
            transforms[joint.child] = result.copy()
        return transforms

    def link_positions(self, positions: Sequence[float]) -> np.ndarray:
        """Return each chain-link origin for conservative swept-path checks."""
        q = np.asarray(positions, dtype=float)
        if q.shape != (len(self.active_names),):
            raise ValueError(f'Expected {len(self.active_names)} joint values, got {q.shape}')
        result = np.eye(4, dtype=float)
        points = []
        for joint in self.joints:
            result = result @ joint.origin
            index = self._active_index.get(joint.name)
            value = 0.0 if index is None else float(q[index])
            if joint.kind in ('revolute', 'continuous'):
                result = result @ _transform(_axis_rotation(joint.axis, value), (0, 0, 0))
            elif joint.kind == 'prismatic':
                result = result @ _transform(np.eye(3), joint.axis * value)
            points.append(result[:3, 3].copy())
        return np.asarray(points, dtype=float)

    @staticmethod
    def pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        translation = target[:3, 3] - current[:3, 3]
        relative = target[:3, :3] @ current[:3, :3].T
        cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        angle = math.acos(cosine)
        skew = np.asarray(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ],
            dtype=float,
        )
        if angle < 1e-7:
            rotation = 0.5 * skew
        elif math.pi - angle < 1e-5:
            # At pi, the ordinary skew/sin formula is singular.  The largest
            # eigenvector of (R + I) / 2 is the rotation axis.  Pick a stable
            # sign so an exact 180-degree mismatch cannot masquerade as zero.
            values, vectors = np.linalg.eigh(0.5 * (relative + np.eye(3)))
            axis = np.asarray(vectors[:, int(np.argmax(values))], dtype=float)
            if np.linalg.norm(skew) > 1e-10:
                if float(np.dot(axis, skew)) < 0.0:
                    axis *= -1.0
            else:
                largest = int(np.argmax(np.abs(axis)))
                if axis[largest] < 0.0:
                    axis *= -1.0
            rotation = angle * axis / max(1e-12, np.linalg.norm(axis))
        else:
            rotation = angle * skew / (2.0 * math.sin(angle))
        return np.concatenate((translation, rotation))

    def _forward_variations(self, positions: Sequence[float]):
        """Build nominal factors for single-coordinate finite differences.

        The baseline forward pose and the solver's finite steps stay unchanged.
        Reuse the prefix and individual remaining factors, preserving the
        original left-to-right multiplication order for every variation. Custom
        forward implementations and ambiguous chain layouts use the ordinary
        solver path. This context exists for only the current iteration.
        """
        if (type(self) is not URDFChain
                or getattr(self.forward, '__func__', None) is not URDFChain.forward):
            return None
        if (any(type(joint) is not Joint for joint in self.joints)
                or len({joint.name for joint in self.joints}) != len(self.joints)
                or len(set(self.active_names)) != len(self.active_names)
                or self._active_index != {
                    name: index for index, name in enumerate(self.active_names)}):
            return None
        movable = {'revolute', 'continuous', 'prismatic'}
        if (any(joint.kind not in movable | {'fixed'} for joint in self.joints)
                or not set(self.active_names) <= {
                    joint.name for joint in self.joints if joint.kind in movable}):
            return None

        q = np.asarray(positions, dtype=float)
        if q.shape != (len(self.active_names),):
            return None
        factors = []
        replacements = {}
        prefix = np.eye(4, dtype=float)
        for joint in self.joints:
            factors.append(joint.origin)
            prefix = prefix @ joint.origin
            index = self._active_index.get(joint.name)
            value = 0.0 if index is None else float(q[index])
            if joint.kind in ('revolute', 'continuous'):
                motion = _transform(_axis_rotation(joint.axis, value), (0, 0, 0))
            elif joint.kind == 'prismatic':
                motion = _transform(np.eye(3), joint.axis * value)
            else:
                continue
            if index is not None:
                replacements[index] = (joint, prefix, len(factors))
            factors.append(motion)
            prefix = prefix @ motion

        def varied(index, value):
            joint, before, factor_index = replacements[index]
            if joint.kind in ('revolute', 'continuous'):
                motion = _transform(_axis_rotation(joint.axis, value), (0, 0, 0))
            else:
                motion = _transform(np.eye(3), joint.axis * value)
            result = before @ motion
            for following in factors[factor_index + 1:]:
                result = result @ following
            return result

        return varied

    def solve(
        self,
        target: np.ndarray,
        seeds: Iterable[Sequence[float]],
        position_tolerance: float = 0.012,
        orientation_tolerance: float = 0.10,
        max_iterations: int = 220,
        damping: float = 0.06,
        fixed_positions: Optional[Mapping[str, float]] = None,
        joint_limit_margin: float = 0.01,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Return a solution, keeping free joints inside their hard limits."""
        if joint_limit_margin < 0.0:
            raise ValueError('joint_limit_margin cannot be negative')
        solve_lower = self.lower + joint_limit_margin
        solve_upper = self.upper - joint_limit_margin
        if np.any(solve_lower > solve_upper):
            raise ValueError('joint_limit_margin leaves an empty joint range')
        best_q: Optional[np.ndarray] = None
        best_score = float('inf')
        weights = np.diag([1.0, 1.0, 1.0, 0.42, 0.42, 0.42])
        fixed_indices: Dict[int, float] = {}
        for name, raw_value in (fixed_positions or {}).items():
            if name not in self._active_index:
                raise ValueError(f'Cannot fix inactive joint {name!r}')
            index = self._active_index[name]
            value = float(raw_value)
            if not self.lower[index] <= value <= self.upper[index]:
                raise ValueError(f'Fixed value for {name!r} is outside its limits')
            fixed_indices[index] = value

        for seed in seeds:
            q = np.clip(np.asarray(seed, dtype=float), solve_lower, solve_upper)
            if q.shape != self.lower.shape:
                continue
            for index, value in fixed_indices.items():
                q[index] = value
            for _ in range(max_iterations):
                current = self.forward(q)
                error = self.pose_error(current, target)
                position_error = float(np.linalg.norm(error[:3]))
                orientation_error = float(np.linalg.norm(error[3:]))
                score = position_error + 0.35 * orientation_error
                if score < best_score:
                    best_q, best_score = q.copy(), score
                if (
                    position_error <= position_tolerance
                    and orientation_error <= orientation_tolerance
                ):
                    return q, score

                jacobian = np.zeros((6, len(q)), dtype=float)
                epsilon = 1e-5
                varied_forward = (
                    self._forward_variations(q)
                    if len(q) - len(fixed_indices) > 1 else None
                )
                for index in range(len(q)):
                    if index in fixed_indices:
                        continue
                    perturbed = q.copy()
                    perturbed[index] = min(
                        solve_upper[index],
                        perturbed[index] + epsilon,
                    )
                    step = perturbed[index] - q[index]
                    if abs(step) < 1e-10:
                        perturbed[index] = max(
                            solve_lower[index],
                            q[index] - epsilon,
                        )
                        step = perturbed[index] - q[index]
                    if abs(step) < 1e-10:
                        continue
                    perturbed_pose = (
                        self.forward(perturbed) if varied_forward is None
                        else varied_forward(index, float(perturbed[index]))
                    )
                    delta = self.pose_error(current, perturbed_pose)
                    jacobian[:, index] = delta / step

                weighted_jacobian = weights @ jacobian
                weighted_error = weights @ error
                system = weighted_jacobian @ weighted_jacobian.T
                update = weighted_jacobian.T @ np.linalg.solve(
                    system + (damping ** 2) * np.eye(6), weighted_error
                )
                q = np.clip(
                    q + np.clip(update, -0.16, 0.16),
                    solve_lower,
                    solve_upper,
                )
                for index, value in fixed_indices.items():
                    q[index] = value

        if best_q is not None:
            best_error = self.pose_error(self.forward(best_q), target)
            if (
                np.linalg.norm(best_error[:3]) <= position_tolerance
                and np.linalg.norm(best_error[3:]) <= orientation_tolerance
            ):
                return best_q, best_score
        return None, best_score


def pose_matrix(position: Sequence[float], rotation: Optional[np.ndarray] = None) -> np.ndarray:
    return _transform(np.eye(3) if rotation is None else rotation, position)


def _box_triangles(size: Sequence[float]) -> np.ndarray:
    half = 0.5 * np.asarray(size, dtype=float)
    if half.shape != (3,) or np.any(half <= 0.0):
        raise ValueError('collision box size must contain three positive values')
    vertices = np.asarray(
        [
            (x * half[0], y * half[1], z * half[2])
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=float,
    )
    faces = np.asarray(
        [
            (0, 1, 3), (0, 3, 2),
            (4, 6, 7), (4, 7, 5),
            (0, 4, 5), (0, 5, 1),
            (2, 3, 7), (2, 7, 6),
            (0, 2, 6), (0, 6, 4),
            (1, 5, 7), (1, 7, 3),
        ],
        dtype=int,
    )
    return vertices[faces]


def _triangles_are_watertight(triangles: np.ndarray) -> bool:
    """Return whether every exact undirected triangle edge occurs twice."""
    surface = np.asarray(triangles, dtype=float)
    if surface.ndim != 3 or surface.shape[1:] != (3, 3) or not len(surface):
        return False
    edges = np.concatenate(
        (
            surface[:, (0, 1)],
            surface[:, (1, 2)],
            surface[:, (2, 0)],
        ),
        axis=0,
    )
    first = edges[:, 0]
    second = edges[:, 1]
    swap = (
        (first[:, 0] > second[:, 0])
        | (
            (first[:, 0] == second[:, 0])
            & (first[:, 1] > second[:, 1])
        )
        | (
            (first[:, 0] == second[:, 0])
            & (first[:, 1] == second[:, 1])
            & (first[:, 2] > second[:, 2])
        )
    )
    canonical = edges.copy()
    canonical[swap] = canonical[swap, ::-1]
    _, counts = np.unique(canonical.reshape((-1, 6)), axis=0, return_counts=True)
    return bool(len(counts) and np.all(counts == 2))


def load_stl_triangles(path: str | Path) -> np.ndarray:
    """Load binary or ASCII STL triangles without an optional mesh package."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack_from('<I', data, 80)[0]
        expected_size = 84 + triangle_count * 50
        if expected_size == len(data):
            record_type = np.dtype(
                [
                    ('normal', '<f4', (3,)),
                    ('vertices', '<f4', (3, 3)),
                    ('attribute', '<u2'),
                ]
            )
            records = np.frombuffer(
                data,
                dtype=record_type,
                count=triangle_count,
                offset=84,
            )
            return np.asarray(records['vertices'], dtype=float)

    vertices = []
    for raw_line in data.decode('ascii').splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == 'vertex':
            vertices.append([float(value) for value in parts[1:]])
    if not vertices or len(vertices) % 3:
        raise ValueError(f'Invalid or unsupported STL file: {path}')
    return np.asarray(vertices, dtype=float).reshape((-1, 3, 3))


def load_urdf_collision_meshes(
    path: str | Path,
    link_names: Iterable[str],
    package_resolver: Callable[[str], str | Path],
    *,
    immutable_local: bool = False,
) -> Tuple[CollisionMesh, ...]:
    """Load selected URDF collision surfaces into their owning link frames."""
    if type(immutable_local) is not bool:
        raise ValueError('immutable_local must be a Boolean')
    if immutable_local:
        from .exact_local_bounds import ModelLocalMesh
    immutable_bytes = immutable_entries = 0
    selected = set(link_names)
    root = ET.parse(str(path)).getroot()
    meshes: List[CollisionMesh] = []
    for link_element in root.findall('link'):
        link_name = link_element.get('name', '')
        if link_name not in selected:
            continue
        for collision in link_element.findall('collision'):
            geometry = collision.find('geometry')
            if geometry is None:
                continue
            triangles: Optional[np.ndarray] = None
            mesh = geometry.find('mesh')
            box = geometry.find('box')
            cylinder = geometry.find('cylinder')
            sphere = geometry.find('sphere')
            if mesh is not None:
                filename = mesh.get('filename', '')
                prefix = 'package://'
                if not filename.startswith(prefix):
                    raise ValueError(f'Unsupported collision mesh URI: {filename!r}')
                package_and_path = filename[len(prefix):].split('/', 1)
                if len(package_and_path) != 2:
                    raise ValueError(f'Invalid package collision mesh URI: {filename!r}')
                package, relative = package_and_path
                triangles = load_stl_triangles(
                    Path(package_resolver(package)) / Path(relative)
                )
                scale = _numbers(mesh.get('scale'), (1.0, 1.0, 1.0))
                triangles = triangles * scale
            elif box is not None:
                triangles = _box_triangles(_numbers(box.get('size'), (0.0, 0.0, 0.0)))
            elif cylinder is not None:
                radius = float(cylinder.get('radius', '0'))
                length = float(cylinder.get('length', '0'))
                triangles = _box_triangles((2.0 * radius, 2.0 * radius, length))
            elif sphere is not None:
                diameter = 2.0 * float(sphere.get('radius', '0'))
                triangles = _box_triangles((diameter, diameter, diameter))
            if triangles is None:
                continue

            origin_element = collision.find('origin')
            xyz = _numbers(
                origin_element.get('xyz') if origin_element is not None else None,
                (0.0, 0.0, 0.0),
            )
            rpy = _numbers(
                origin_element.get('rpy') if origin_element is not None else None,
                (0.0, 0.0, 0.0),
            )
            rotation = _rpy_matrix(rpy)
            local_triangles = triangles @ rotation.T + xyz
            bounds = np.asarray(
                [
                    np.min(local_triangles, axis=(0, 1)),
                    np.max(local_triangles, axis=(0, 1)),
                ]
            )
            watertight = _triangles_are_watertight(local_triangles)
            local_mesh = None
            if (immutable_local and immutable_entries < ModelLocalMesh.MAXIMUM_ENTRIES
                    and immutable_bytes + local_triangles.nbytes + 48 <= ModelLocalMesh.MAXIMUM_BYTES):
                # Only this model-producer opt-in freezes geometry. Unsupported
                # surfaces retain the previous raw-array path and semantics.
                try:
                    local_mesh = ModelLocalMesh(local_triangles)
                except ValueError:
                    pass
                if local_mesh is not None:
                    local_triangles = local_mesh.snapshot()[0]
                    immutable_bytes += local_mesh.nbytes
                    immutable_entries += 1
            meshes.append(
                CollisionMesh(
                    link_name,
                    local_triangles,
                    bounds,
                    watertight,
                    local_mesh,
                )
            )

    loaded_links = {mesh.link for mesh in meshes}
    missing = selected - loaded_links
    if missing:
        raise ValueError(f'URDF collision geometry missing for links: {sorted(missing)}')
    return tuple(meshes)


def oriented_box_from_corners(
    corners: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center, column axes, and half extents for ordered box corners."""
    points = np.asarray(corners, dtype=float)
    if points.shape != (8, 3):
        raise ValueError('oriented box must contain eight 3-D corners')
    edge_indices = (4, 2, 1)
    edges = np.asarray([points[index] - points[0] for index in edge_indices])
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError('oriented box edges must have positive length')
    axes = (edges / lengths[:, None]).T
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-7):
        raise ValueError('oriented box edges must be orthogonal')
    return np.mean(points, axis=0), axes, 0.5 * lengths


def oriented_box_intersects_triangles(
    corners: Sequence[Sequence[float]],
    triangles: Sequence[Sequence[Sequence[float]]],
    tolerance: float = 1e-9,
    *,
    closed_surface: bool = False,
) -> bool:
    """Return whether an oriented box intersects any triangle via SAT tests."""
    center, axes, half_extents = oriented_box_from_corners(corners)
    surface = np.asarray(triangles, dtype=float)
    if surface.size == 0:
        return False
    if surface.ndim != 3 or surface.shape[1:] != (3, 3):
        raise ValueError('collision surface must have shape (N, 3, 3)')

    # Transform each triangle into the box frame, making the tested box AABB.
    vertices = (surface - center) @ axes
    separated = np.any(
        (np.max(vertices, axis=1) < -half_extents - tolerance)
        | (np.min(vertices, axis=1) > half_extents + tolerance),
        axis=1,
    )
    # A facet rejected on a box axis cannot intersect the box. Keep only the
    # remaining facets for the more expensive SAT axes, preserving their order
    # and the exact comparisons above. Retain the complete ``surface`` for the
    # closed-mesh containment test below: a contained box may touch no facets.
    vertices = vertices[~separated]
    # If no facets remain, skip only empty SAT work;
    # a closed mesh can still contain the box, so retain ray parity below.
    if len(vertices):
        separated = np.zeros(len(vertices), dtype=bool)

        edge_a = vertices[:, 1] - vertices[:, 0]
        edge_b = vertices[:, 2] - vertices[:, 1]
        edge_c = vertices[:, 0] - vertices[:, 2]
        normals = np.cross(edge_a, -edge_c)
        plane_distance = np.abs(np.einsum('ij,ij->i', normals, vertices[:, 0]))
        plane_radius = np.abs(normals) @ half_extents
        separated |= plane_distance > plane_radius + tolerance

        basis = np.eye(3, dtype=float)
        for edge in (edge_a, edge_b, edge_c):
            for box_axis in basis:
                test_axis = np.cross(edge, box_axis)
                projections = np.einsum('nij,nj->ni', vertices, test_axis)
                radius = np.abs(test_axis) @ half_extents
                separated |= (
                    (np.max(projections, axis=1) < -radius - tolerance)
                    | (np.min(projections, axis=1) > radius + tolerance)
                )
        if np.any(~separated):
            return True
    if not closed_surface:
        return False

    # Triangle SAT detects surface crossings and meshes contained by the box,
    # but not a box wholly contained by a larger closed mesh.  A ray parity
    # test from the box centre covers that remaining containment case.
    direction = np.asarray([1.0, 0.3713906764, 0.6947465906], dtype=float)
    direction /= np.linalg.norm(direction)
    edge_a = surface[:, 1] - surface[:, 0]
    edge_b = surface[:, 2] - surface[:, 0]
    cross_direction = np.cross(direction, edge_b)
    determinant = np.einsum('ij,ij->i', edge_a, cross_direction)
    usable = np.abs(determinant) > tolerance
    inverse = np.zeros_like(determinant)
    inverse[usable] = 1.0 / determinant[usable]
    offset = center - surface[:, 0]
    barycentric_u = np.einsum('ij,ij->i', offset, cross_direction) * inverse
    cross_offset = np.cross(offset, edge_a)
    barycentric_v = np.einsum('j,ij->i', direction, cross_offset) * inverse
    distance = np.einsum('ij,ij->i', edge_b, cross_offset) * inverse
    hit_distances = np.sort(
        distance[
            usable
            & (barycentric_u >= -tolerance)
            & (barycentric_v >= -tolerance)
            & (barycentric_u + barycentric_v <= 1.0 + tolerance)
            & (distance > tolerance)
        ]
    )
    if not len(hit_distances):
        return False
    distinct_hits = 1 + int(
        np.count_nonzero(np.diff(hit_distances) > max(tolerance, 1e-7))
    )
    return bool(distinct_hits % 2)


def triangles_intersect(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
    tolerance: float = 1e-9,
) -> bool:
    """Return whether two closed 3-D triangles touch or intersect.

    The separating-axis set contains both face normals, every edge cross
    product, and in-plane edge normals.  The latter are required when the
    triangles are coplanar, where the ordinary 3-D edge-cross axes collapse
    onto the shared face normal.
    """
    triangle_a = np.asarray(first, dtype=float)
    triangle_b = np.asarray(second, dtype=float)
    if triangle_a.shape != (3, 3) or triangle_b.shape != (3, 3):
        raise ValueError('triangles must each have shape (3, 3)')
    if tolerance < 0.0:
        raise ValueError('tolerance cannot be negative')
    if not np.all(np.isfinite(triangle_a)) or not np.all(np.isfinite(triangle_b)):
        raise ValueError('triangle coordinates must be finite')

    edges_a = np.roll(triangle_a, -1, axis=0) - triangle_a
    edges_b = np.roll(triangle_b, -1, axis=0) - triangle_b
    normal_a = np.cross(edges_a[0], edges_a[1])
    normal_b = np.cross(edges_b[0], edges_b[1])
    return _triangles_intersect_with_basis(
        triangle_a, triangle_b, edges_a, edges_b, normal_a, normal_b, tolerance,
    )


def _triangles_intersect_with_basis(
    triangle_a, triangle_b, edges_a, edges_b, normal_a, normal_b, tolerance,
):
    """Internal SAT kernel: callers supply validated exact same-surface bases."""
    # Coplanar triangles also need the three 2-D edge-normal axes.  Include
    # both triangles' axes so the test remains symmetric for degenerate input.
    # Construct later groups only if earlier axes did not separate the pair.
    # Keep the exact original axis order and cross-product arithmetic; scalar
    # normalization, projection comparisons and degeneracy handling below are
    # unchanged. Most separated facets need only the two face normals.
    def axis_groups():
        yield (normal_a, normal_b)
        yield np.cross(edges_a[:, None, :], edges_b[None, :, :]).reshape(9, 3)
        yield np.cross(normal_a, edges_a)
        yield np.cross(normal_b, edges_b)

    usable_axis = False
    for axes in axis_groups():
        for axis in axes:
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                continue
            usable_axis = True
            direction = axis / norm
            projection_a = triangle_a @ direction
            projection_b = triangle_b @ direction
            if (
                float(np.max(projection_a)) < float(np.min(projection_b)) - tolerance
                or float(np.max(projection_b))
                < float(np.min(projection_a)) - tolerance
            ):
                return False

    # Collision meshes should not contain zero-area facets.  Treat two fully
    # degenerate triangles as intersecting only when their bounds overlap;
    # this is conservative for a safety guard.
    if not usable_axis:
        return bool(
            np.all(
                np.max(triangle_a, axis=0) + tolerance
                >= np.min(triangle_b, axis=0)
            )
            and np.all(
                np.max(triangle_b, axis=0) + tolerance
                >= np.min(triangle_a, axis=0)
            )
        )
    return True


def _mesh_components(mesh: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Split by exact shared vertices; preserve first-triangle component order."""
    if not len(mesh):
        return ()
    # Exactly the existing equality/first-owner mapping: no coordinate rounding,
    # welding, relative-frame transformation or approximate equivalence.
    _, first_vertex, inverse = np.unique(
        mesh.reshape(-1, 3), axis=0, return_index=True, return_inverse=True,
    )
    owners = first_vertex[inverse] // 3
    triangles = np.repeat(np.arange(len(mesh), dtype=np.intp), 3)
    linked = owners != triangles
    graph = csr_matrix(
        (np.ones(np.count_nonzero(linked), dtype=bool),
         (triangles[linked], owners[linked])),
        shape=(len(mesh), len(mesh)), dtype=bool,
    )
    # Each old union(i, owner) is precisely an undirected graph edge. Do not
    # rely on SciPy's label numbering for the observable component ordering.
    _, labels = connected_components(graph, directed=False, return_labels=True)
    order = np.argsort(labels, kind='stable')
    boundaries = np.flatnonzero(np.diff(labels[order])) + 1
    groups = list(np.split(order, boundaries))
    groups.sort(key=lambda indices: int(indices[0]))
    return tuple(mesh[indices] for indices in groups)


def _point_inside_watertight_mesh(
    point: np.ndarray,
    mesh: np.ndarray,
    tolerance: float,
) -> bool:
    """Classify a point by majority ray parity against a closed surface."""
    directions = np.asarray(
        [
            (1.0, 0.3713906764, 0.6947465906),
            (0.2134131557, 1.0, 0.5176380902),
            (0.6134895274, 0.2791839741, 1.0),
        ],
        dtype=float,
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    edge_a = mesh[:, 1] - mesh[:, 0]
    edge_b = mesh[:, 2] - mesh[:, 0]
    offset = point - mesh[:, 0]
    votes = 0
    distinct_tolerance = max(tolerance, 1e-7)
    for direction in directions:
        cross_direction = np.cross(direction, edge_b)
        determinant = np.einsum('ij,ij->i', edge_a, cross_direction)
        usable = np.abs(determinant) > tolerance
        inverse = np.zeros_like(determinant)
        inverse[usable] = 1.0 / determinant[usable]
        barycentric_u = np.einsum(
            'ij,ij->i', offset, cross_direction
        ) * inverse
        cross_offset = np.cross(offset, edge_a)
        barycentric_v = np.einsum(
            'j,ij->i', direction, cross_offset
        ) * inverse
        distance = np.einsum('ij,ij->i', edge_b, cross_offset) * inverse
        hit_distances = np.sort(
            distance[
                usable
                & (barycentric_u >= -tolerance)
                & (barycentric_v >= -tolerance)
                & (barycentric_u + barycentric_v <= 1.0 + tolerance)
                & (distance > tolerance)
            ]
        )
        if len(hit_distances):
            distinct_hits = 1 + int(
                np.count_nonzero(
                    np.diff(hit_distances) > distinct_tolerance
                )
            )
            votes += int(distinct_hits % 2)
    return votes >= 2


def _watertight_components_contain_component(
    container_components: Sequence[np.ndarray],
    candidate_components: Sequence[np.ndarray],
    tolerance: float,
) -> bool:
    """Return whether a candidate component lies inside a closed mesh."""
    candidate_points = [component[0, 0] for component in candidate_components]
    for container_component in container_components:
        lower = np.min(container_component, axis=(0, 1))
        upper = np.max(container_component, axis=(0, 1))
        for point in candidate_points:
            if np.any(point < lower - tolerance) or np.any(
                point > upper + tolerance
            ):
                continue
            if _point_inside_watertight_mesh(
                point,
                container_component,
                tolerance,
            ):
                return True
    return False


class PreparedTriangleMesh:
    """An immutable world-space surface with lazy, exact pair-test data.

    Reuse only this snapshot, never a nearby pose or a nominal finger aperture.
    The owned byte buffer prevents later input mutation from invalidating the
    cached bounds or exact connected components. No rounding, welding, frame
    change or collision verdict is cached here.
    """

    __slots__ = ('_surface', '_bounds', '_facet_bounds', '_components', '_facet_basis')

    def __init__(self, surface: Sequence[Sequence[Sequence[float]]]):
        mesh = np.asarray(surface, dtype=float)
        if mesh.size:
            if mesh.ndim != 3 or mesh.shape[1:] != (3, 3):
                raise ValueError('collision surface must have shape (N, 3, 3)')
            if not np.all(np.isfinite(mesh)):
                raise ValueError('collision surface coordinates must be finite')
        self._surface = self._immutable(mesh)
        self._bounds = None
        self._facet_bounds = None
        self._components = None
        self._facet_basis = None

    @staticmethod
    def _immutable(array: np.ndarray) -> np.ndarray:
        return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)

    @property
    def surface(self) -> np.ndarray:
        return self._surface

    @property
    def bounds(self) -> np.ndarray:
        if self._bounds is None:
            self._bounds = self._immutable(np.asarray([
                np.min(self._surface, axis=(0, 1)),
                np.max(self._surface, axis=(0, 1)),
            ]))
        return self._bounds

    @property
    def facet_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._facet_bounds is None:
            self._facet_bounds = (
                self._immutable(np.min(self._surface, axis=1)),
                self._immutable(np.max(self._surface, axis=1)),
            )
        return self._facet_bounds

    @property
    def facet_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Exact edges and face normals for this immutable world snapshot."""
        if self._facet_basis is None:
            edges = np.roll(self._surface, -1, axis=1) - self._surface
            normals = np.cross(edges[:, 0, :], edges[:, 1, :])
            self._facet_basis = (self._immutable(edges), self._immutable(normals))
        return self._facet_basis

    @property
    def components(self) -> Tuple[np.ndarray, ...]:
        if self._components is None:
            self._components = tuple(
                self._immutable(component) for component in _mesh_components(self._surface)
            )
        return self._components


def triangle_meshes_intersect(
    first: Sequence[Sequence[Sequence[float]]] | PreparedTriangleMesh,
    second: Sequence[Sequence[Sequence[float]]] | PreparedTriangleMesh,
    tolerance: float = 1e-9,
    *,
    first_watertight: bool = False,
    second_watertight: bool = False,
) -> bool:
    """Return whether two triangle meshes intersect or one contains the other.

    Mesh and per-facet axis-aligned bounds provide a cheap broad phase before
    the exact triangle SAT test.  Set the corresponding ``*_watertight`` flag
    only for a verified closed surface to enable containment detection.  This
    function deliberately treats touching surfaces as a collision. Prepared
    inputs reuse exact world-surface data across pairs; raw inputs retain the
    same validation, arithmetic, triangle order and containment behavior.
    """
    prepared_a = first if isinstance(first, PreparedTriangleMesh) else None
    prepared_b = second if isinstance(second, PreparedTriangleMesh) else None
    mesh_a = prepared_a.surface if prepared_a is not None else np.asarray(first, dtype=float)
    mesh_b = prepared_b.surface if prepared_b is not None else np.asarray(second, dtype=float)
    for mesh, prepared in ((mesh_a, prepared_a), (mesh_b, prepared_b)):
        if prepared is not None:
            continue  # Constructor validated the immutable snapshot.
        if mesh.size == 0:
            continue
        if mesh.ndim != 3 or mesh.shape[1:] != (3, 3):
            raise ValueError('collision surface must have shape (N, 3, 3)')
        if not np.all(np.isfinite(mesh)):
            raise ValueError('collision surface coordinates must be finite')
    if tolerance < 0.0:
        raise ValueError('tolerance cannot be negative')
    if mesh_a.size == 0 or mesh_b.size == 0:
        return False

    bounds_a = prepared_a.bounds if prepared_a is not None else np.asarray(
        [np.min(mesh_a, axis=(0, 1)), np.max(mesh_a, axis=(0, 1))]
    )
    bounds_b = prepared_b.bounds if prepared_b is not None else np.asarray(
        [np.min(mesh_b, axis=(0, 1)), np.max(mesh_b, axis=(0, 1))]
    )
    if np.any(bounds_a[1] < bounds_b[0] - tolerance) or np.any(
        bounds_b[1] < bounds_a[0] - tolerance
    ):
        return False

    # Remove facets whose AABB is disjoint from the other *whole* mesh before
    # choosing the smaller loop side.  Such a facet cannot participate in a
    # triangle intersection.  Keep the complete meshes below for watertight
    # containment: slicing them there could change connected components and
    # their representative points, which would change collision semantics.
    minimum_a, maximum_a = (prepared_a.facet_bounds if prepared_a is not None else
                            (np.min(mesh_a, axis=1), np.max(mesh_a, axis=1)))
    minimum_b, maximum_b = (prepared_b.facet_bounds if prepared_b is not None else
                            (np.min(mesh_b, axis=1), np.max(mesh_b, axis=1)))
    candidates_a = (
        np.all(maximum_a + tolerance >= bounds_b[0], axis=1)
        & np.all(bounds_b[1] + tolerance >= minimum_a, axis=1)
    )
    candidates_b = (
        np.all(maximum_b + tolerance >= bounds_a[0], axis=1)
        & np.all(bounds_a[1] + tolerance >= minimum_b, axis=1)
    )
    narrow_a = mesh_a[candidates_a]
    narrow_minimum_a = minimum_a[candidates_a]
    narrow_maximum_a = maximum_a[candidates_a]
    narrow_b = mesh_b[candidates_b]
    narrow_minimum_b = minimum_b[candidates_b]
    narrow_maximum_b = maximum_b[candidates_b]
    narrow_prepared_a, narrow_prepared_b = prepared_a, prepared_b
    indices_a, indices_b = np.flatnonzero(candidates_a), np.flatnonzero(candidates_b)

    # Preserve the original full-mesh length decision, including triangle
    # argument order, then loop only its filtered facets.  The unchanged
    # triangle SAT remains the narrow phase and still treats touching as
    # collision.
    if len(mesh_a) > len(mesh_b):
        narrow_prepared_a, narrow_prepared_b = narrow_prepared_b, narrow_prepared_a
        indices_a, indices_b = indices_b, indices_a
        narrow_a, narrow_b = narrow_b, narrow_a
        narrow_minimum_a, narrow_minimum_b = (
            narrow_minimum_b,
            narrow_minimum_a,
        )
        narrow_maximum_a, narrow_maximum_b = (
            narrow_maximum_b,
            narrow_maximum_a,
        )
    for source_index_a, triangle_a, triangle_minimum_a, triangle_maximum_a in zip(
        indices_a,
        narrow_a,
        narrow_minimum_a,
        narrow_maximum_a,
        strict=True,
    ):
        candidates = np.flatnonzero(
            np.all(
                triangle_maximum_a + tolerance >= narrow_minimum_b,
                axis=1,
            )
            & np.all(
                narrow_maximum_b + tolerance >= triangle_minimum_a,
                axis=1,
            )
        )
        for index in candidates:
            if narrow_prepared_a is not None and narrow_prepared_b is not None:
                edges_a, normals_a = narrow_prepared_a.facet_basis
                edges_b, normals_b = narrow_prepared_b.facet_basis
                source_index_b = indices_b[index]
                intersects = _triangles_intersect_with_basis(
                    triangle_a, narrow_b[index],
                    edges_a[source_index_a], edges_b[source_index_b],
                    normals_a[source_index_a], normals_b[source_index_b], tolerance,
                )
            else:
                intersects = triangles_intersect(triangle_a, narrow_b[index], tolerance)
            if intersects:
                return True
    if not first_watertight and not second_watertight:
        return False
    components_a = prepared_a.components if prepared_a is not None else _mesh_components(mesh_a)
    components_b = prepared_b.components if prepared_b is not None else _mesh_components(mesh_b)
    return bool(
        first_watertight
        and _watertight_components_contain_component(
            components_a,
            components_b,
            tolerance,
        )
        or second_watertight
        and _watertight_components_contain_component(
            components_b,
            components_a,
            tolerance,
        )
    )


def interpolate_joint_waypoints(
    start: Sequence[float],
    end: Sequence[float],
    *,
    first_arm_index: int,
    maximum_joint_step: float,
    orientation_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    maximum_orientation_step: Optional[float] = None,
    maximum_segments: int = 256,
) -> Tuple[np.ndarray, ...]:
    """Split a joint move into bounded waypoints, excluding the start.

    When an orientation-distance callback is supplied, the subdivision is
    refined until every adjacent grasp-link attitude change is also bounded.
    """
    first = np.asarray(start, dtype=float)
    last = np.asarray(end, dtype=float)
    if first.ndim != 1 or last.shape != first.shape:
        raise ValueError('joint endpoints must be equal-length vectors')
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(last)):
        raise ValueError('joint endpoints must be finite')
    if not 0 <= first_arm_index < len(first):
        raise ValueError('first_arm_index is outside the joint vector')
    if maximum_joint_step <= 0.0:
        raise ValueError('maximum_joint_step must be positive')
    if (orientation_distance is None) != (maximum_orientation_step is None):
        raise ValueError(
            'orientation_distance and maximum_orientation_step must be used together'
        )
    if maximum_orientation_step is not None and maximum_orientation_step <= 0.0:
        raise ValueError('maximum_orientation_step must be positive')
    if maximum_segments < 1:
        raise ValueError('maximum_segments must be positive')

    delta = last - first
    arm_delta = np.abs(delta[first_arm_index:])
    minimum_segments = max(
        1,
        int(math.ceil(float(np.max(arm_delta)) / maximum_joint_step)),
    )
    for segment_count in range(minimum_segments, maximum_segments + 1):
        waypoints = tuple(
            first + delta * (index / segment_count)
            for index in range(1, segment_count + 1)
        )
        if orientation_distance is None:
            return waypoints
        previous = first
        bounded = True
        for waypoint in waypoints:
            distance = float(orientation_distance(previous, waypoint))
            if not math.isfinite(distance) or distance > maximum_orientation_step:
                bounded = False
                break
            previous = waypoint
        if bounded:
            return waypoints
    raise RuntimeError('joint transition exceeds the subdivision limit')
