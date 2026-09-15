"""Owned ROS-free geometry for an isolated PLACE evaluation process.

This module performs no subscription, action, clock, process creation or live
admission. A future parent dispatcher must capture and admit each query and
retain every existing pre-command scene/contact/freshness check.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import threading
from types import MappingProxyType
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import (
    CollisionMesh, PreparedTriangleMesh, URDFChain,
    load_stl_triangles, load_urdf_collision_meshes, oriented_box_intersects_triangles,
    triangle_meshes_intersect,
)
from .motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS
from .runtime_utils import joint_state_cache_key, measured_joint_positions


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _finite_bytes(value, count, name):
    current = np.asarray(value, dtype=np.float64)
    if current.shape != (count,) or not np.isfinite(current).all():
        raise ValueError(name + ' must contain finite values')
    return current.tobytes(order='C')


def _exact_joint_state_cache_key(*groups):
    """Owned backend only: cache current finite values without rounding.

    The existing node keeps its original 10-decimal key. Exact keys here make
    a body/static result independent of which nearby query a worker saw first.
    """
    try:
        values = tuple(float(value) for group in groups for value in group)
    except (TypeError, ValueError) as exc:
        raise ValueError('collision cache state must be numeric') from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError('collision cache state must be finite')
    return struct.pack(f'={len(values)}d', *values)


class _BoundedExactCache(dict):
    """FIFO exact-result storage; eviction only forces a fresh computation."""
    def __init__(self, maximum_entries):
        if type(maximum_entries) is not int or maximum_entries <= 0:
            raise ValueError('positive exact-cache capacity required')
        super().__init__()
        self.maximum_entries = maximum_entries

    def __setitem__(self, key, value):
        if key not in self and len(self) >= self.maximum_entries:
            del self[next(iter(self))]
        super().__setitem__(key, value)


@dataclass(frozen=True)
class GeometryQuery:
    """Owned exact sample input; producer admission remains a parent duty."""
    request_id: int
    epoch: int
    source_id: str
    model_id: str
    scene_id: str
    q: bytes
    right: bytes
    head: bytes
    aperture: float
    loaded: bool

    @property
    def input_sha256(self):
        return _canonical_digest(dict(request_id=self.request_id, epoch=self.epoch,
            source_id=self.source_id, model_id=self.model_id, scene_id=self.scene_id,
            q=self.q.hex(), right=self.right.hex(), head=self.head.hex(),
            aperture=np.asarray(self.aperture, dtype=np.float64).tobytes().hex(),
            loaded=self.loaded, byteorder=sys.byteorder))

    @classmethod
    def capture(cls, *, request_id, epoch, source_id, model_id, scene_id,
                q, right, head, aperture, loaded):
        if type(request_id) is not int or request_id < 0 or type(epoch) is not int or epoch < 0:
            raise ValueError('request identity must be nonnegative integers')
        if any(type(v) is not str or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
               for v in (source_id, model_id, scene_id)):
            raise ValueError('geometry identities must be SHA256 values')
        if type(loaded) is not bool or isinstance(aperture, bool) or not math.isfinite(aperture):
            raise ValueError('invalid aperture or loaded flag')
        return cls(request_id, epoch, source_id, model_id, scene_id,
                   _finite_bytes(q, 8, 'q'), _finite_bytes(right, 7, 'right'),
                   _finite_bytes(head, 2, 'head'), float(aperture), loaded)

    def validated(self):
        if type(self) is not GeometryQuery:
            raise ValueError('exact geometry query required')
        if type(self.q) is not bytes or type(self.right) is not bytes or type(self.head) is not bytes:
            raise ValueError('query vectors must be owned bytes')
        if (len(self.q), len(self.right), len(self.head)) != (64, 56, 16):
            raise ValueError('query byte lengths are invalid')
        fresh = type(self).capture(request_id=self.request_id, epoch=self.epoch,
            source_id=self.source_id, model_id=self.model_id, scene_id=self.scene_id,
            q=np.frombuffer(self.q, dtype=np.float64),
            right=np.frombuffer(self.right, dtype=np.float64),
            head=np.frombuffer(self.head, dtype=np.float64),
            aperture=self.aperture, loaded=self.loaded)
        if fresh != self:
            raise ValueError('query changed during validation')
        return fresh


@dataclass(frozen=True)
class GeometryReply:
    request_id: int
    epoch: int
    source_id: str
    model_id: str
    scene_id: str
    input_sha256: str
    verdict: bool
    rejection_json: bytes
    owner_prefix_minimum: Optional[float]
    owner_samples: int
    owner_cache_hits: int


class PinnedGeometryAssets:
    """Verify exact source and every referenced collision asset before loading.

    Paths are supplied by the parent from its admitted installation manifest;
    neither filesystem search nor package-index discovery selects a model.
    """
    def __init__(self, *, source_root, source_files, urdf, bin_mesh, packages, asset_files):
        self.source_root = Path(source_root).resolve()
        self.source_files = MappingProxyType(dict(source_files))
        self.urdf, self.bin_mesh = Path(urdf).resolve(), Path(bin_mesh).resolve()
        self.packages = MappingProxyType({name: Path(path).resolve() for name, path in packages.items()})
        description = self.packages['erc_description']
        self.table_mesh = description / 'models/table/meshes/erc_base_table.STL'
        self.table_sdf = description / 'models/table/sdf/erc_table.sdf'
        self.asset_files = MappingProxyType({str(Path(path).resolve()): pin for path, pin in asset_files.items()})
        self.source_id = _canonical_digest(dict(self.source_files))
        self.model_id = _canonical_digest(dict(urdf=str(self.urdf), bin_mesh=str(self.bin_mesh),
            packages={k: str(v) for k, v in self.packages.items()}, files=dict(self.asset_files)))
        self.verify()
        self._sealed = True

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('pinned geometry assets are immutable')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError('pinned geometry assets are immutable')

    def verify(self):
        if not self.source_files or not self.asset_files:
            raise ValueError('empty source or model manifest')
        actual = {p.relative_to(self.source_root).as_posix(): _digest(p)
                  for p in self.source_root.rglob('*') if p.is_file()
                  and not {'__pycache__', '.pytest_cache', '.ruff_cache'}.intersection(p.parts)
                  and p.suffix != '.pyc'}
        if actual != dict(self.source_files):
            raise ValueError('complete geometry source inventory mismatch')
        for relative, pin in self.source_files.items():
            path = (self.source_root / relative).resolve()
            if not path.is_relative_to(self.source_root) or _digest(path) != pin:
                raise ValueError('geometry source pin mismatch: ' + str(relative))
        if Path(__file__).resolve() != self.source_root / 'erc_phase1_solution/pure_geometry_owner.py':
            raise ValueError('geometry owner import origin mismatch')
        for name, module in tuple(sys.modules.items()):
            if name.startswith('erc_phase1_solution') and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                if not path.is_relative_to(self.source_root):
                    raise ValueError('mixed geometry source origin: ' + name)
                if path.relative_to(self.source_root).as_posix() not in self.source_files:
                    raise ValueError('unmanifested geometry import: ' + name)
        for path, pin in self.asset_files.items():
            if _digest(path) != pin:
                raise ValueError('geometry asset pin mismatch: ' + path)
        required = {str(self.urdf), str(self.bin_mesh), str(self.table_mesh), str(self.table_sdf)}
        for mesh in ET.parse(self.urdf).getroot().findall('./link/collision/geometry/mesh'):
            uri = mesh.get('filename', '')
            if not uri.startswith('package://'):
                raise ValueError('unsupported collision mesh URI')
            package, relative = uri[len('package://'):].split('/', 1)
            root = self.packages[package]
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError('collision mesh escapes package')
            required.add(str(path))
        if not required.issubset(self.asset_files):
            raise ValueError('collision assets absent from model manifest')

    def verify_scene_models(self, registered_bin):
        """Repeat the original scene-construction model/registration gates."""
        from .table_scene import TABLE_MESH_SHA256
        if _digest(self.bin_mesh) != registered_bin['mesh_sha256']:
            raise RuntimeError('placement_scene_bin_model_changed')
        if (_digest(self.table_mesh) != TABLE_MESH_SHA256
                or _digest(self.table_sdf) not in (
                    '90f3aa39affdadcf6ade532ed19a7898d84b813478791b38f40e2316e5d07f4f',
                    '04f1940665775cdff35cdc89ac3518c5db3cbb4337891d9a601b8906c857417f')):
            raise RuntimeError('placement_scene_table_model_changed')


# Collision link order and adjacency exclusions below are the unchanged
# production definitions. They are ordinary source, not an evaluated facade.


LEFT_COLLISION_LINKS = tuple(
    f'arm_left_{index}_link' for index in range(1, 8)
)


RIGHT_COLLISION_LINKS = tuple(
    f'arm_right_{index}_link' for index in range(1, 8)
)


HEAD_COLLISION_LINKS = (
    'head_1_link',
    'head_2_link',
    'head_front_camera_link',
)


HEAD_JOINTS = ('head_1_joint', 'head_2_joint')


CARRIED_COLLISION_LINKS = (
    'base_link',
    'torso_base_link',
    'torso_lift_link',
    *LEFT_COLLISION_LINKS,
    *RIGHT_COLLISION_LINKS,
    *HEAD_COLLISION_LINKS,
)


DIRECTLY_CONNECTED_COLLISION_LINKS = {
    frozenset(('base_link', 'torso_base_link')),
    frozenset(('torso_base_link', 'torso_lift_link')),
    frozenset(('torso_lift_link', 'arm_left_1_link')),
    frozenset(('torso_lift_link', 'arm_right_1_link')),
    frozenset(('torso_lift_link', 'head_1_link')),
    frozenset(('head_1_link', 'head_2_link')),
    frozenset(('head_2_link', 'head_front_camera_link')),
    *(
        frozenset((f'arm_left_{index}_link', f'arm_left_{index + 1}_link'))
        for index in range(1, 7)
    ),
    *(
        frozenset((f'arm_right_{index}_link', f'arm_right_{index + 1}_link'))
        for index in range(1, 7)
    ),
}


class RobotCollisionGeometry:
    """Shared pure body-geometry methods; owner supplies model/state fields."""

    def _collision_link_transforms(
        self,
        solution: Sequence[float],
        *,
        right_positions: Optional[Sequence[float]] = None,
        head_positions: Optional[Sequence[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return moving-left, parked-right, and measured-head transforms."""
        q = np.asarray(solution, dtype=float)
        transforms = self.chain.link_transforms(q)
        resolved_right = self._resolved_right_positions(right_positions)
        right_q = np.concatenate(([q[0]], resolved_right))
        right_transforms = self.right_chain.link_transforms(right_q)
        transforms.update(
            {
                link: right_transforms[link]
                for link in RIGHT_COLLISION_LINKS
            }
        )
        head_q = np.concatenate(
            ([q[0]], self._resolved_head_positions(head_positions))
        )
        head_transforms = self.head_chain.link_transforms(head_q)
        transforms.update(
            {link: head_transforms[link] for link in HEAD_COLLISION_LINKS}
        )
        return transforms


    def _right_joint_positions(self) -> np.ndarray:
        """Return the measured parked-arm joints or fail closed."""
        return np.asarray(
            measured_joint_positions(
                self.joints,
                RIGHT_ARM_JOINTS,
                group='right arm',
            ),
            dtype=float,
        )


    def _resolved_right_positions(
        self,
        positions: Optional[Sequence[float]],
    ) -> np.ndarray:
        result = (
            self._right_joint_positions()
            if positions is None
            else np.asarray(positions, dtype=float)
        )
        if (
            result.shape != (len(RIGHT_ARM_JOINTS),)
            or not np.all(np.isfinite(result))
        ):
            raise ValueError(
                'right-arm collision state must contain seven finite joints'
            )
        return result


    def _head_joint_positions(self) -> np.ndarray:
        """Return measured head joints, failing closed when state is absent."""
        missing = [name for name in HEAD_JOINTS if name not in self.joints]
        if missing:
            raise RuntimeError(
                f'Cannot collision-check without head joints: {missing}'
            )
        return np.asarray([float(self.joints[name]) for name in HEAD_JOINTS])


    def _resolved_head_positions(
        self,
        positions: Optional[Sequence[float]],
    ) -> np.ndarray:
        result = (
            self._head_joint_positions()
            if positions is None
            else np.asarray(positions, dtype=float)
        )
        if result.shape != (len(HEAD_JOINTS),) or not np.all(np.isfinite(result)):
            raise ValueError('head collision state must contain two finite joints')
        return result


    def _world_collision_surfaces(
        self,
        solution: Sequence[float],
        *,
        right_positions: Optional[Sequence[float]] = None,
        head_positions: Optional[Sequence[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Transform and group official robot collision facets by link."""
        transforms = self._collision_link_transforms(
            solution,
            right_positions=right_positions,
            head_positions=head_positions,
        )
        grouped: Dict[str, List[np.ndarray]] = {}
        for collision_mesh in self.carried_collision_meshes:
            transform = transforms[collision_mesh.link]
            world_triangles = (
                collision_mesh.triangles @ transform[:3, :3].T
                + transform[:3, 3]
            )
            grouped.setdefault(collision_mesh.link, []).append(world_triangles)
        return {
            link: meshes[0] if len(meshes) == 1 else np.concatenate(meshes)
            for link, meshes in grouped.items()
        }


    def _watertight_collision_links(self) -> frozenset[str]:
        """Return links whose complete collision surface is verified closed."""
        closed_by_link: Dict[str, bool] = {}
        for collision_mesh in getattr(self, 'carried_collision_meshes', ()):
            closed_by_link[collision_mesh.link] = (
                closed_by_link.get(collision_mesh.link, True)
                and collision_mesh.watertight
            )
        return frozenset(
            link for link, closed in closed_by_link.items() if closed
        )


    def _robot_self_collision(
        self,
        solution: Sequence[float],
        collision_surfaces: Optional[Dict[str, np.ndarray]] = None,
        *,
        head_positions: Optional[Sequence[float]] = None,
        right_positions: Optional[Sequence[float]] = None,
        _scene_snapshot=None,
    ) -> Optional[Tuple[str, str]]:
        """Return the first intersecting pair of nonadjacent robot links."""
        q = np.asarray(solution, dtype=float)
        right_positions = self._resolved_right_positions(right_positions)
        resolved_head = self._resolved_head_positions(head_positions)
        cache_key = _exact_joint_state_cache_key(
            q,
            right_positions,
            resolved_head,
        )
        if collision_surfaces is None and cache_key in self._self_collision_cache:
            return self._self_collision_cache[cache_key]
        shared_geometry = None
        if collision_surfaces is None and _scene_snapshot is not None:
            from erc_phase1_solution.sample_collision_snapshot import _SceneRobotSnapshot
            if (type(_scene_snapshot) is _SceneRobotSnapshot
                    and getattr(self._world_collision_surfaces, '__func__', None)
                        is RobotCollisionGeometry._world_collision_surfaces
                    and getattr(self._collision_link_transforms, '__func__', None)
                        is RobotCollisionGeometry._collision_link_transforms):
                shared_geometry = _scene_snapshot.resolve(self, q, right_positions, resolved_head)
        surfaces = (
            self._world_collision_surfaces(
                q,
                right_positions=right_positions,
                head_positions=resolved_head,
            )
            if collision_surfaces is None
            else collision_surfaces
        ) if shared_geometry is None else shared_geometry[0]
        links = [link for link in CARRIED_COLLISION_LINKS if link in surfaces]
        watertight_links = self._watertight_collision_links()
        bounds = {
            link: np.asarray(
                [
                    np.min(surface, axis=(0, 1)),
                    np.max(surface, axis=(0, 1)),
                ]
            )
            for link, surface in surfaces.items()
        } if shared_geometry is None else shared_geometry[1]
        prepared_surfaces = {}

        def links_intersect(first_link: str, second_link: str) -> bool:
            first_bounds = bounds[first_link]
            second_bounds = bounds[second_link]
            if np.any(first_bounds[1] < second_bounds[0]) or np.any(
                second_bounds[1] < first_bounds[0]
            ):
                return False
            if collision_surfaces is None:
                separator = getattr(self, '_static_pair_separation', None)
                if separator is None:
                    from erc_phase1_solution.static_pair_separation import StaticPairSeparation
                    separator = StaticPairSeparation()
                    self._static_pair_separation = separator
                if separator.separated(
                    first_link, second_link,
                    surfaces[first_link], surfaces[second_link],
                ):
                    return False
            for link in (first_link, second_link):
                if link not in prepared_surfaces:
                    prepared_surfaces[link] = PreparedTriangleMesh(surfaces[link])
            return triangle_meshes_intersect(
                prepared_surfaces[first_link],
                prepared_surfaces[second_link],
                first_watertight=first_link in watertight_links,
                second_watertight=second_link in watertight_links,
            )

        static_links = [
            link for link in links if link not in LEFT_COLLISION_LINKS
        ]
        static_cache_key = _exact_joint_state_cache_key(
            (q[0],),
            right_positions,
            resolved_head,
        )
        static_result = None
        if collision_surfaces is None:
            static_result = self._static_self_collision_cache.get(
                static_cache_key
            )
        if static_result is None and (
            collision_surfaces is not None
            or static_cache_key not in self._static_self_collision_cache
        ):
            for first_index, first_link in enumerate(static_links):
                for second_link in static_links[first_index + 1:]:
                    if frozenset((first_link, second_link)) in (
                        DIRECTLY_CONNECTED_COLLISION_LINKS
                    ):
                        continue
                    if links_intersect(first_link, second_link):
                        static_result = (first_link, second_link)
                        break
                if static_result is not None:
                    break
            if collision_surfaces is None:
                self._static_self_collision_cache[
                    static_cache_key
                ] = static_result
        if static_result is not None:
            if collision_surfaces is None:
                self._self_collision_cache[cache_key] = static_result
            return static_result

        for first_index, first_link in enumerate(links):
            for second_link in links[first_index + 1:]:
                if (
                    first_link not in LEFT_COLLISION_LINKS
                    and second_link not in LEFT_COLLISION_LINKS
                ):
                    continue
                if frozenset((first_link, second_link)) in (
                    DIRECTLY_CONNECTED_COLLISION_LINKS
                ):
                    continue
                if links_intersect(first_link, second_link):
                    result = (first_link, second_link)
                    if collision_surfaces is None:
                        self._self_collision_cache[cache_key] = result
                    return result
        if collision_surfaces is None:
            self._self_collision_cache[cache_key] = None
        return None


    def _carried_robot_collision(
        self,
        solution: Sequence[float],
        world_corners: Sequence[Sequence[float]],
        *,
        head_positions: Optional[Sequence[float]] = None,
        maximum_robot_x: Optional[float] = None,
        right_positions: Optional[Sequence[float]] = None,
    ) -> Optional[str]:
        """Return the first robot link intersecting the inflated carried book."""
        transforms = self._collision_link_transforms(
            solution,
            head_positions=head_positions,
            **({'right_positions': right_positions} if right_positions is not None else {}),
        )
        book_bounds = np.asarray(
            [
                np.min(world_corners, axis=0),
                np.max(world_corners, axis=0),
            ]
        )
        for collision_mesh in self.carried_collision_meshes:
            transform = transforms[collision_mesh.link]
            local_center = np.mean(collision_mesh.bounds, axis=0)
            local_half_extents = 0.5 * (
                collision_mesh.bounds[1] - collision_mesh.bounds[0]
            )
            world_center = (
                transform[:3, :3] @ local_center + transform[:3, 3]
            )
            world_half_extents = (
                np.abs(transform[:3, :3]) @ local_half_extents
            )
            if (
                maximum_robot_x is not None
                and float(world_center[0] + world_half_extents[0])
                > maximum_robot_x
            ):
                return collision_mesh.link
            if np.any(
                world_center + world_half_extents < book_bounds[0]
            ) or np.any(
                world_center - world_half_extents > book_bounds[1]
            ):
                continue
            world_triangles = (
                collision_mesh.triangles @ transform[:3, :3].T
                + transform[:3, 3]
            )
            if oriented_box_intersects_triangles(
                world_corners,
                world_triangles,
                closed_surface=collision_mesh.watertight,
            ):
                return collision_mesh.link
        return None


    _scene_snapshot_body = _robot_self_collision


class RobotGeometryOwner(RobotCollisionGeometry):
    """One process owns real chains, immutable models and ordinary caches.

    No object from a live ROS node is stored. The shared base retains the
    original predicate order and uses exact body/static cache keys.
    """
    maximum_queries = 20_000
    maximum_distinct_apertures = 128
    def __init__(self, assets):
        from .shelf_cradle_geometry import ShelfCradleGeometry
        from .installed_geometry_identity import InstalledGeometryIdentity
        if type(assets) not in (PinnedGeometryAssets, InstalledGeometryIdentity):
            raise ValueError('exact pinned or installed geometry assets required')
        assets.verify()
        self.assets = assets
        urdf = assets.urdf
        self.chain = URDFChain.from_urdf(urdf, 'base_footprint', 'gripper_left_grasping_link', IK_JOINTS)
        self.right_chain = URDFChain.from_urdf(urdf, 'base_footprint', 'gripper_right_grasping_link',
                                             ('torso_lift_joint', *RIGHT_ARM_JOINTS))
        self.head_chain = URDFChain.from_urdf(urdf, 'base_footprint', 'head_front_camera_link',
                                            ('torso_lift_joint', *HEAD_JOINTS))
        self.carried_collision_meshes = load_urdf_collision_meshes(
            urdf, CARRIED_COLLISION_LINKS, assets.packages.__getitem__, immutable_local=True)
        self._shelf_cradle_geometry = ShelfCradleGeometry(urdf, assets.packages.__getitem__, immutable_local=True)
        bin_triangles = load_stl_triangles(assets.bin_mesh)
        # Owned bytes prevent later changes to constructor inputs from changing
        # the obstacle geometry used by this owner.
        self._bin_triangles = np.frombuffer(bin_triangles.tobytes(), dtype=np.float64).reshape(bin_triangles.shape)
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self.joints = {}
        self._self_collision_cache = _BoundedExactCache(8192)
        self._static_self_collision_cache = _BoundedExactCache(4096)
        self._scene = None
        self._epoch = -1
        self._last_request_id = -1
        self._queries_seen = 0
        self._apertures = set()
        assets.verify()

    def begin_scene(self, *, epoch, scene):
        from .scene_checked_place import (
            NominalBinObstacle, PlaceSceneChecker, ScreenEnvelope,
            TableSceneObstacle, certified_bin_cavity,
        )
        from .bin_scene_admission import validate_bin_scene
        if type(epoch) is not int or epoch <= self._epoch:
            raise ValueError('scene epoch must advance')
        fields={'bin_floor_point', 'bin_scene', 'table_scene', 'attached_corners', 'transition_samples'}
        if type(scene) is not dict or set(scene) not in (fields, fields | {'carried_volume_context'}):
            raise ValueError('scene fields are incomplete or unexpected')
        current = json.loads(json.dumps(scene, sort_keys=True, allow_nan=False))
        count = current['transition_samples']
        if type(count) is not int or count < 3:
            raise ValueError('transition sample count must be at least three')
        attached = np.asarray(current['attached_corners'], dtype=np.float64)
        if attached.shape != (8, 3) or not np.isfinite(attached).all():
            raise ValueError('held corners must contain eight finite points')
        registered = validate_bin_scene(current['bin_scene'])
        if type(self.assets) is PinnedGeometryAssets:
            self.assets.verify_scene_models(registered)
        else:
            self.assets.verify_scene_models(registered, current['table_scene'])
        triangles = self._bin_triangles
        obstacle = NominalBinObstacle(current['bin_floor_point'],
            [triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))],
            cavity_bounds=certified_bin_cavity(triangles), bin_scene=registered)
        table = TableSceneObstacle(current['table_scene'])
        screen = ScreenEnvelope(self.assets.urdf)
        self.carried_transition_samples = count
        self._held_book_corners = np.frombuffer(attached.tobytes(), dtype=np.float64).reshape(8, 3)
        checker = PlaceSceneChecker(self, obstacle, self._shelf_cradle_geometry,
                                    self._held_book_corners, table=table, screen=screen)
        if checker._world_geometry_cache is None:
            raise RuntimeError('ordinary owned geometry cache was not admitted')
        if 'carried_volume_context' in current:
            from .place_volume_geometry import validate_volume_context, CONTEXT_FIELDS
            context=validate_volume_context(current['carried_volume_context'])
            for key,name in CONTEXT_FIELDS.items():
                setattr(self,name,context[key])
            checker._place_volume_context=context
        self._scene = checker
        self._scene_id = _canonical_digest(current)
        self._epoch = epoch
        self._last_request_id = -1
        self._queries_seen = 0
        self._apertures.clear()
        # Body/static caches belong to this fixed model owner and persist across
        # scene invocations. This backend uses exact keys, so nearby states do
        # not acquire the first rounded-key verdict seen by another sample.
        self._cancel.clear()
        return self._scene_id

    def cancel(self):
        """Sticky until a new explicit epoch; parent still discards late replies."""
        self._cancel.set()

    def evaluate_independent(self, query):
        """Evaluate without committing scene history from a speculative suffix.

        Exact model/body caches remain private to this owner. Scene success
        caches and diagnostics are parent-owned: each reply describes only the
        current query, including a minimum lowered before a later rejection.
        One caller at a time is required, as for evaluate/begin_scene.
        """
        from .geometry_process_protocol import SampleDelta
        from .place_volume_geometry import PlaceVolumeQuery, evaluate_owned_volume
        if type(query) is PlaceVolumeQuery:
            return evaluate_owned_volume(self,query)
        scene = self._scene
        if scene is None:
            raise RuntimeError('no active scene')
        table = scene.table
        sentinel = object()
        saved = (scene.minimum_moving_left_z, scene.last_rejection, scene.cache,
                 scene.samples, scene.cache_hits,
                 None if table is None else table.last_intersection)
        scene.minimum_moving_left_z = math.inf
        scene.last_rejection = None
        scene.cache = {}
        scene.samples = scene.cache_hits = 0
        if table is not None:
            table.last_intersection = sentinel
        try:
            reply = self.evaluate(query)
            touched = table is not None and table.last_intersection is not sentinel
            return SampleDelta(query.request_id, query.epoch, query.source_id,
                query.model_id, query.scene_id, query.input_sha256, reply.verdict,
                reply.owner_prefix_minimum, json.loads(reply.rejection_json),
                touched, table.last_intersection if touched else None).validate()
        finally:
            (scene.minimum_moving_left_z, scene.last_rejection, scene.cache,
             scene.samples, scene.cache_hits, previous_table) = saved
            if table is not None:
                table.last_intersection = previous_table

    def evaluate(self, query):
        if type(query) is not GeometryQuery:
            raise ValueError('exact geometry query required')
        query.validated()
        if self._scene is None:
            raise RuntimeError('no active scene')
        if (query.epoch != self._epoch or query.source_id != self.assets.source_id
                or query.model_id != self.assets.model_id or query.scene_id != self._scene_id):
            raise RuntimeError('geometry query identity mismatch')
        if query.request_id <= self._last_request_id:
            raise RuntimeError('geometry queries must arrive in increasing order')
        if (self._queries_seen >= self.maximum_queries
                or (query.aperture not in self._apertures
                    and len(self._apertures) >= self.maximum_distinct_apertures)):
            self._cancel.set()
            raise RuntimeError('geometry owner resource bound exceeded')
        self._queries_seen += 1
        self._apertures.add(query.aperture)
        self._last_request_id = query.request_id
        with self._lock:
            self.joints = dict(zip(RIGHT_ARM_JOINTS, np.frombuffer(query.right, dtype=np.float64)))
            self.joints.update(zip(HEAD_JOINTS, np.frombuffer(query.head, dtype=np.float64)))
        scene = self._scene
        try:
            verdict = scene.sample(np.frombuffer(query.q, dtype=np.float64), query.aperture, query.loaded)
        except BaseException:
            self._cancel.set()
            raise
        # A worker cannot authorize movement. Cancellation arriving after an
        # expensive sample invalidates this reply as well as subsequent work.
        if self._cancel.is_set():
            verdict = scene._reject('cancelled')
        minimum = float(scene.minimum_moving_left_z)
        if math.isnan(minimum) or minimum == -math.inf:
            raise RuntimeError('invalid geometry minimum')
        return GeometryReply(query.request_id, query.epoch, query.source_id,
            query.model_id, query.scene_id, query.input_sha256, bool(verdict),
            json.dumps(scene.last_rejection, sort_keys=True, allow_nan=False).encode(),
            None if minimum == math.inf else minimum, scene.samples, scene.cache_hits)
