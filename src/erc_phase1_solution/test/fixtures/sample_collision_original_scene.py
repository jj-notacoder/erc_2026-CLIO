"""Nominal scene-checked, supported PLACE planning; no commands or ROS queries.

Registered mode uses the admitted camera-fitted CAD pose and its measurement
allowance. The legacy mode retains its floor-reference/approach-ray assumption.
Certified empty cavity space is removed from
conservative bin material with a 5 mm margin. Optional camera registration adds
the finite tabletop and legs. The head screen visual box is a planner obstacle.
Official nominal mimic meshes do not describe actual finger deflection. Checks
are sampled and the falling released book is not dynamically modeled. Existing
retention, cancellation and measured release remain separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import itertools
import math
from pathlib import Path
import time

import numpy as np

from .kinematics import (
    PreparedTriangleMesh, _box_triangles, load_stl_triangles,
    oriented_box_intersects_triangles, triangle_meshes_intersect,
)
from .motion_profiles import HOME, OFFER, PREGRASP
from .scene_cartesian_solver import SearchLimits, solve_scene_cartesian
from .rigid_palm_preflight import PALM_COLLISION_LINK
from .shelf_cradle_geometry import ShelfCradleGeometry
from .table_scene import TableSceneObstacle, TABLE_MESH_SHA256
from .empty_pickup_collision import ScreenEnvelope
from .bin_scene_admission import validate_bin_scene
from .static_pair_separation import separated_on_axes
from .exact_local_bounds import ExactLocalBounds

_BOX_CORNER_TRIANGLES = (((_box_triangles([2., 2., 2.])+1.)/2)
                         @ np.asarray([4., 2., 1.])).astype(int)


def _finite(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f'invalid {label}')
    return result.copy()


class NominalBinObstacle:
    """Conservative CAD material at the visual floor point and approach ray.

    Without a certified cavity this is a solid outer box. With a cavity it is
    five separately closed wall/floor slabs, each expanded by the margin. A
    union is never passed to a parity containment test as one closed mesh.
    """

    def __init__(self, point, outer_bounds, *, margin=.005, cavity_bounds=None, bin_scene=None):
        self.point = _finite(point, (3,), 'bin point')
        raw = _finite(outer_bounds, (2, 3), 'bin bounds')
        if np.any(raw[1] <= raw[0]) or not math.isfinite(margin) or margin < 0:
            raise ValueError('invalid bin extent or margin')
        self.registered_scene = None
        if bin_scene is None:
            ray = np.r_[self.point[:2], 0.]
            if np.linalg.norm(ray) < 1e-9:
                raise ValueError('bin point has no horizontal approach ray')
            ray /= np.linalg.norm(ray)
            self.rotation = np.column_stack(([-ray[1], ray[0], 0.], [0., 0., 1.], ray))
            # Legacy floor-reference assumption; selected registered mode below
            # never substitutes a ray or patch point for a failed measured pose.
            self.origin = self.point - self.rotation @ np.asarray([0., -.095, 0.])
        else:
            registered = validate_bin_scene(bin_scene)
            if (not np.allclose(raw, registered['outer_bounds'], rtol=0., atol=1e-6)
                    or np.linalg.norm(self.point-np.asarray(registered['floor_center'])) > 1e-6
                    or cavity_bounds is None
                    or not np.allclose(cavity_bounds, registered['cavity_bounds'], rtol=0., atol=1e-6)):
                raise ValueError('registered bin pose does not match the selected CAD and target')
            self.rotation = np.asarray(registered['rotation'], dtype=float)
            self.origin = np.asarray(registered['origin'], dtype=float)
            margin = max(margin, registered['modeled_margin_m']) + registered['registration_margin_m']
            self.registered_scene = registered
        self.bounds = raw + np.asarray([[-margin]*3, [margin]*3])
        centre = self.bounds.mean(axis=0)
        half = (self.bounds[1] - self.bounds[0]) / 2
        signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
        self.corners = (centre + signs*half) @ self.rotation.T + self.origin
        self.triangles = (_box_triangles(2*half) + centre) @ self.rotation.T + self.origin
        self.margin = float(margin)
        self.material_bounds = [self.bounds]
        self.cavity_bounds = None
        if cavity_bounds is not None:
            cavity = _finite(cavity_bounds, (2, 3), 'bin cavity')
            if (np.any(cavity[1] <= cavity[0]) or np.any(cavity[0] < raw[0])
                    or np.any(cavity[1] > raw[1]+1e-8)
                    or not np.isclose(cavity[1, 1], raw[1, 1], atol=1e-8, rtol=0.)):
                raise ValueError('cavity must be inside the bounds and open at the top')
            self.cavity_bounds = cavity
            slabs = []
            floor = raw.copy(); floor[1, 1] = cavity[0, 1]; slabs.append(floor)
            for axis in (0, 2):
                low = raw.copy(); low[1, axis] = cavity[0, axis]; slabs.append(low)
                high = raw.copy(); high[0, axis] = cavity[1, axis]; slabs.append(high)
            self.material_bounds = [slab + [[-margin]*3, [margin]*3] for slab in slabs]
        self.material_corners = []
        self.material_triangles = []
        for slab in self.material_bounds:
            centre, half = slab.mean(axis=0), (slab[1]-slab[0])/2
            self.material_corners.append((centre+signs*half) @ self.rotation.T+self.origin)
            self.material_triangles.append((_box_triangles(2*half)+centre) @ self.rotation.T+self.origin)

        self._local_bounds = ExactLocalBounds()

    def intersects(self, surface, transform, watertight):
        """Conservative local AABB broad phase, unchanged SAT/containment narrow phase."""
        cache = getattr(self, '_local_bounds', None)
        cached = (cache.capture(surface) if cache is not None
                  else ExactLocalBounds.model_snapshot(surface))
        if cached is None:
            low, high = surface.min(axis=(0, 1)), surface.max(axis=(0, 1))
        else:
            surface, low, high = cached
        rotation = self.rotation.T @ transform[:3, :3]
        translation = self.rotation.T @ (transform[:3, 3] - self.origin)
        centre = rotation @ ((low + high)/2) + translation
        half = np.abs(rotation) @ ((high-low)/2)
        if np.any(centre+half < self.bounds[0]) or np.any(centre-half > self.bounds[1]):
            return False
        world = surface @ transform[:3, :3].T + transform[:3, 3]
        return any(oriented_box_intersects_triangles(corners, world, closed_surface=watertight)
                   for corners, slab in zip(self.material_corners, self.material_bounds)
                   if not (np.any(centre+half < slab[0]) or np.any(centre-half > slab[1])))

    def book_intersects(self, corners):
        return any(oriented_box_intersects_triangles(corners, surface, closed_surface=True)
                   for surface in self.material_triangles)


def certified_bin_cavity(triangles):
    """Use the known rectangular void only if all CAD faces remain outside it.

    Triangle AABBs are conservative: a face crossing this inner core disables
    the void even if its vertices are outside. The numerical tolerance is much
    smaller than the modeled 5 mm material expansion.
    """
    cavity = np.asarray([[-.145, -.095, -.245], [.145, .105, .245]])
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    if np.any(np.all(high > cavity[0]+1e-6, axis=1)
              & np.all(low < cavity[1]-1e-6, axis=1)):
        return None
    bounds = np.asarray([triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))])
    if (np.any(cavity[0] < bounds[0]) or np.any(cavity[1] > bounds[1]+1e-8)
            or not np.isclose(cavity[1, 1], bounds[1, 1], atol=1e-8, rtol=0.)):
        return None
    return cavity


def coordinated_supported_goals(chain, start, goal, *, margin, minimum_support,
                                shoulder_progress_power=1., joint_progress_powers=None,
                                proximal_steps=20):
    """Bounded proximal states with nearby legal q7 maximizing SIGNED jaw-up.

    The endpoint remains the exact original IK solution. These are candidate
    goals only: the existing controller subdivision and full transition/support
    checks must subsequently accept every interpolation.
    """
    first = _finite(start, (8,), 'coordinator start')
    last = _finite(goal, (8,), 'coordinator goal')
    if (not math.isfinite(margin) or margin < 0
            or not math.isfinite(minimum_support) or not 0 < minimum_support <= 1
            or shoulder_progress_power not in (1., .5)
            or type(proximal_steps) is not int or proximal_steps not in (10, 20)):
        raise ValueError('invalid coordinator policy')
    powers = (np.asarray([shoulder_progress_power, 1., 1., 1., 1., 1.])
              if joint_progress_powers is None else
              _finite(joint_progress_powers, (6,), 'joint progress powers'))
    if not all(value in (.5, 1., 2.) for value in powers):
        raise ValueError('unsupported joint progress power')
    lower, upper = np.asarray(chain.lower), np.asarray(chain.upper)
    for q in (first, last):
        if (np.any(q < lower) or np.any(q > upper)
                or np.any(q[1:] < lower[1:]+margin)
                or np.any(q[1:] > upper[1:]-margin)):
            return None
    lo, hi = float(lower[-1]+margin), float(upper[-1]-margin)
    if lo > hi:
        return None
    previous = float(first[-1])
    goals = []
    for fraction in np.linspace(1./proximal_steps, 1., proximal_steps):
        q = first + (last-first)*fraction
        q[1:7] = first[1:7] + (last[1:7]-first[1:7]) * fraction**powers
        zero = q.copy(); zero[-1] = 0.
        quarter = zero.copy(); quarter[-1] = np.pi/2
        optimum = math.atan2(float(chain.forward(quarter)[2, 1]), float(chain.forward(zero)[2, 1]))
        options = [lo, hi]
        options.extend(optimum+k*2*np.pi for k in range(-2, 3) if lo <= optimum+k*2*np.pi <= hi)
        scored = []
        for roll in options:
            probe = q.copy(); probe[-1] = roll
            scored.append((float(chain.forward(probe)[2, 1]), float(roll)))
        maximum = max(support for support, _ in scored)
        if not math.isfinite(maximum) or maximum < minimum_support:
            return None
        roll = min((roll for support, roll in scored if maximum-support <= 1e-9),
                   key=lambda value: (abs(value-previous), value))
        if abs(roll-previous) > np.pi:
            return None
        q[-1] = roll
        goals.append(q); previous = roll
    if abs(float(last[-1])-previous) > np.pi:
        return None
    if not np.array_equal(goals[-1], last):
        goals.append(last.copy())
    return goals


class _SampleWorldAabbs:
    """Lazy reductions of private world arrays, discarded after one sample.

    Only remember fresh outputs of the ordinary NumPy transforms below. The
    source arrays may be mutable: the owned output shares no storage with
    them. Subclasses, custom operations and aliased/nonstandard outputs take
    the original reduction path. This is not a cache for caller-owned arrays.
    Current collision predicates only read these private outputs.
    """

    def __init__(self):
        self._entries = {}

    @staticmethod
    def _plain(value):
        return type(value) is np.ndarray and value.dtype == np.dtype(float)

    @classmethod
    def _owned_world(cls, value):
        return (cls._plain(value) and value.ndim == 3 and value.shape[1:] == (3, 3)
                and len(value) > 0 and value.base is None and value.flags.owndata
                and value.flags.c_contiguous)

    def remember_transform(self, world, source, transform):
        if self._plain(source) and self._plain(transform) and self._owned_world(world):
            self._entries[id(world)] = (world, {})

    def merge(self, values):
        if len(values) == 1:
            return values[0]
        world = np.concatenate(values)
        if (self._owned_world(world)
                and all(self._entries.get(id(value), (None,))[0] is value for value in values)):
            self._entries[id(world)] = (world, {})
        return world

    def reduce(self, surface, operation):
        entry = self._entries.get(id(surface))
        if entry is None or entry[0] is not surface:
            return getattr(surface, operation)(axis=(0, 1))
        values = entry[1]
        if operation not in values:
            values[operation] = getattr(surface, operation)(axis=(0, 1))
        return values[operation]


class PlaceSceneChecker:
    """One planning invocation; exact sample/context cache, no route reuse later."""

    def __init__(self, node, obstacle, tool_geometry, attached_corners, table=None, screen=None):
        self.node, self.obstacle, self.tool = node, obstacle, tool_geometry
        self.table = table
        self.screen = screen
        self.attached = _finite(attached_corners, (8, 3), 'held corners')
        self.cache = {}
        self.samples = self.cache_hits = 0
        self.minimum_moving_left_z = math.inf
        self.last_rejection = None

    def _reject(self, reason, **details):
        self.last_rejection = dict(reason=reason, **details)
        return False

    def sample(self, q, aperture, loaded):
        node = self.node
        if node._cancel.is_set():
            return self._reject('cancelled')
        q = _finite(q, (8,), 'scene joints')
        if not math.isfinite(aperture):
            raise ValueError('nonfinite scene aperture')
        # Take right/head together. Cache identity uses exact bytes and does not
        # round poses or mix geometry from different measured parked contexts.
        lock = getattr(node, '_lock', None)
        if lock is None:
            right, head = node._resolved_right_positions(None), node._resolved_head_positions(None)
        else:
            with lock:
                right, head = node._resolved_right_positions(None), node._resolved_head_positions(None)
        context = dict(right_positions=right, head_positions=head)
        key = (q.tobytes(), float(aperture), bool(loaded), right.tobytes(), head.tobytes())
        self.samples += 1
        if key in self.cache:
            self.cache_hits += 1
            return True
        transforms = node._collision_link_transforms(q, **context)
        world_aabbs = _SampleWorldAabbs()
        robot = {}
        for mesh in node.carried_collision_meshes:
            transform = transforms[mesh.link]
            surface = mesh.triangles
            bin_surface = table_surface = surface
            local_mesh = getattr(mesh, 'local_mesh', None)
            if local_mesh is not None:
                from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
                if type(local_mesh) is ModelLocalMesh and local_mesh.matches(surface):
                    # Custom obstacle implementations keep their original raw
                    # array call signature. Only exact built-ins opt in.
                    if type(self.obstacle) is NominalBinObstacle:
                        bin_surface = local_mesh
                    if type(self.table) is TableSceneObstacle:
                        table_surface = local_mesh
            if self.obstacle.intersects(bin_surface, transform, mesh.watertight):
                return self._reject('robot_bin', link=mesh.link)
            if self.table is not None and self.table.intersects(table_surface, transform, mesh.watertight):
                return self._reject('robot_table', link=mesh.link, solid=self.table.last_intersection)
            world = surface @ transform[:3, :3].T + transform[:3, 3]
            world_aabbs.remember_transform(world, surface, transform)
            robot.setdefault(mesh.link, []).append(world)
        robot = {link: world_aabbs.merge(values) for link, values in robot.items()}
        hand = node.chain.forward(q)
        tool = {}
        for link, surface in self.tool.local_surfaces(aperture).items():
            if self.obstacle.intersects(surface, hand, self.tool.watertight[link]):
                return self._reject('tool_bin', link=link)
            if self.table is not None and self.table.intersects(surface, hand, self.tool.watertight[link]):
                return self._reject('tool_table', link=link, solid=self.table.last_intersection)
            tool[link] = surface @ hand[:3, :3].T + hand[:3, 3]
            world_aabbs.remember_transform(tool[link], surface, hand)
        if self.screen is not None:
            screen = self.screen.world(q[0], head)
            low, high = screen.min(axis=0), screen.max(axis=0)
            closed = node._watertight_collision_links()
            for link, surface in {**robot, **tool}.items():
                if link in ('head_1_link', 'head_2_link', 'head_front_camera_link'):
                    continue
                if (np.any(world_aabbs.reduce(surface, 'max') < low)
                        or np.any(high < world_aabbs.reduce(surface, 'min'))):
                    continue
                if oriented_box_intersects_triangles(screen, surface,
                        closed_surface=link in closed or self.tool.watertight.get(link, False)):
                    return self._reject('head_screen', link=link)
        if loaded:
            book = self.attached @ hand[:3, :3].T + hand[:3, 3]
            if self.obstacle.book_intersects(book):
                return self._reject('book_bin')
            if self.table is not None and self.table.intersects_box(book):
                return self._reject('book_table', solid=self.table.last_intersection)
            if self.screen is not None and oriented_box_intersects_triangles(
                    book, screen[_BOX_CORNER_TRIANGLES], closed_surface=True):
                return self._reject('book_head_screen')
            self.minimum_moving_left_z = min(self.minimum_moving_left_z, float(book[:, 2].min()))
        # Reuse the cradle checker's exact tool/robot pair rule, including its
        # wrist/palm adjacency exclusion. It applies to open return too, without
        # keeping a fictional released book attached. Tool/tool and intended
        # finger/book contacts are not additional pairs in this policy.
        # Supplying custom surfaces bypasses the planner's existing parked-link
        # caches; keep its normal context-aware call and use our surfaces below.
        collision = node._robot_self_collision(q, **context)
        if collision is not None:
            return self._reject('robot_self', pair=list(collision))
        bounds = {link: np.asarray([world_aabbs.reduce(surface, 'min'),
                                    world_aabbs.reduce(surface, 'max')])
                  for link, surface in robot.items()}
        closed = node._watertight_collision_links()
        prepared_robot, prepared_tool = {}, {}
        for link, bound in bounds.items():
            if link.startswith('arm_left_'):
                self.minimum_moving_left_z = min(self.minimum_moving_left_z, float(bound[0, 2]))
                if bound[0, 2] < .02:
                    return self._reject('left_arm_ground', link=link)
        for link, surface in tool.items():
            low, high = world_aabbs.reduce(surface, 'min'), world_aabbs.reduce(surface, 'max')
            self.minimum_moving_left_z = min(self.minimum_moving_left_z, float(low[2]))
            if low[2] < .02:
                return self._reject('tool_ground', link=link)
            for other, other_surface in robot.items():
                if link == PALM_COLLISION_LINK and other == 'arm_left_7_link':
                    continue
                other_low, other_high = bounds[other]
                if np.any(high < other_low) or np.any(other_high < low):
                    continue
                # Use the hand axes only as directions to try. The proof
                # reprojects both complete current world meshes; it cannot
                # reuse a nominal rigid-pair verdict or ignore a collision.
                if separated_on_axes(surface, other_surface, hand[:3, :3].T):
                    continue
                if link not in prepared_tool:
                    prepared_tool[link] = PreparedTriangleMesh(surface)
                if other not in prepared_robot:
                    prepared_robot[other] = PreparedTriangleMesh(other_surface)
                if triangle_meshes_intersect(prepared_tool[link], prepared_robot[other],
                        first_watertight=self.tool.watertight[link], second_watertight=other in closed):
                    return self._reject('tool_robot', tool_link=link, robot_link=other)
        self.cache[key] = True
        return True

    def leg(self, first, last, aperture, loaded):
        first, last = np.asarray(first), np.asarray(last)
        count = (1 if np.array_equal(first, last) else max(
            3, int(self.node.carried_transition_samples),
            int(math.ceil(float(np.max(np.abs(last-first)))/.02))+1))
        for fraction in np.linspace(0., 1., count):
            if not self.sample(first+(last-first)*fraction, aperture, loaded):
                self.last_rejection['fraction'] = float(fraction)
                return False
        return True


@dataclass
class SceneCheckedPlacePlan:
    solutions: list
    orientation_index: int
    path_score: float
    setup: list
    unloaded_home: list
    diagnostics: dict
    empty_return_from_clearance: list | None = None


def plan_scene_checked_place(node, positions, rotation, carried_start, torso_ready,
                             staging_seed, bin_point, resolve_package, *, table_scene=None, bin_scene=None):
    """Build all motion and empty-return checks before returning any plan.

    Stored staging is an IK initial guess only. Physical setup starts at the
    supplied measured carry state through the checked torso segment. Existing
    global retracted planning remains solely on the future unloaded HOME fold.
    """
    started = time.monotonic()

    def progress(stage, **fields):
        # Selected table-scene planning only; telemetry cannot change its result.
        try:
            publisher = getattr(node, '_publish_status', None)
            if table_scene is None or not callable(publisher):
                return
            fields = deepcopy(fields)
            for name in ('bin_floor_point', 'actual_carry_start', 'staging_seed',
                         'torso_ready', 'cartesian_positions', 'tool_rotation',
                         'held_book_corners'):
                if name in fields:
                    fields[name] = np.asarray(fields[name]).tolist()
            if 'master_producer_stamp_ns' in fields:
                fields['master_producer_stamp_ns'] = int(fields['master_producer_stamp_ns'])
            if 'search_wall_budget_seconds' in fields:
                fields['search_wall_budget_seconds'] = float(fields['search_wall_budget_seconds'])
            publisher('place_planning_stage', command='place', stage=stage,
                      planning_wall_seconds=time.monotonic()-started,
                      **fields)
        except Exception:
            pass

    if node._cancel.is_set():
        raise RuntimeError('scene_checked_place_cancelled')
    feedback, error = node._adaptive_motion_feedback()
    if error is not None or feedback is None:
        raise RuntimeError(f'scene_checked_place_feedback:{error}')
    aperture = float(feedback.position)  # Raw accepted measurement, not pad gap.
    open_aperture = float(node.gripper_open)
    if not math.isfinite(open_aperture) or not 0 <= open_aperture <= .069:
        raise ValueError('invalid configured opening position')
    try:
        progress('scene_construction_begin',
                 selected_bin_scene=bin_scene, selected_table_scene=table_scene,
                 selected_admission_reference=getattr(node, '_selected_place_scene_reference', None),
                 bin_floor_point=bin_point, actual_carry_start=carried_start,
                 staging_seed=staging_seed, torso_ready=torso_ready,
                 cartesian_positions=positions, tool_rotation=rotation,
                 held_book_corners=node._held_book_corners,
                 measured_master=aperture, master_producer_stamp_ns=feedback.stamp_ns)
    except Exception:
        pass
    share = Path(resolve_package('erc_description'))
    urdf = share / 'urdf' / 'tiago_pro.urdf'
    bin_mesh = share / 'models' / 'collection_bin' / 'meshes' / 'erc_base_collection_bin.STL'
    if getattr(node, 'bin_scene_required', False) and bin_scene is None:
        raise RuntimeError('scene_checked_place_bin_pose_required')
    if bin_scene is not None:
        bin_scene = validate_bin_scene(bin_scene)
        if hashlib.sha256(bin_mesh.read_bytes()).hexdigest() != bin_scene['mesh_sha256']:
            raise RuntimeError('placement_scene_bin_model_changed')
    triangles = load_stl_triangles(bin_mesh)
    obstacle = NominalBinObstacle(bin_point, [triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))],
                                  cavity_bounds=certified_bin_cavity(triangles), bin_scene=bin_scene)
    tool = getattr(node, '_shelf_cradle_geometry', None)
    if tool is None:
        tool = ShelfCradleGeometry(urdf, resolve_package)
        node._shelf_cradle_geometry = tool
    table = None
    if table_scene is not None:
        table_path = share / 'models' / 'table' / 'meshes' / 'erc_base_table.STL'
        table_sdf = share / 'models' / 'table' / 'sdf' / 'erc_table.sdf'
        if (hashlib.sha256(table_path.read_bytes()).hexdigest() != TABLE_MESH_SHA256
                or hashlib.sha256(table_sdf.read_bytes()).hexdigest()
                    not in ('90f3aa39affdadcf6ade532ed19a7898d84b813478791b38f40e2316e5d07f4f',
                            '04f1940665775cdff35cdc89ac3518c5db3cbb4337891d9a601b8906c857417f')):
            # These two audited SDF byte forms differ only in line endings.
            raise RuntimeError('placement_scene_table_model_changed')
        table = TableSceneObstacle(table_scene)
    scene = PlaceSceneChecker(node, obstacle, tool, node._held_book_corners, table=table,
                              screen=ScreenEnvelope(urdf))
    progress('scene_construction_completed')
    attached = scene.attached
    progress('torso_validation_begin')
    if (not node._carried_robot_transition_is_safe(carried_start, torso_ready, attached)
            or not node._gravity_supported_transition_is_safe(carried_start, torso_ready)
            or not scene.leg(carried_start, torso_ready, aperture, True)):
        raise RuntimeError(f'scene_checked_place_torso_rejected:{scene.last_rejection}')
    progress('torso_validation_completed', scene_samples=scene.samples)
    # The supplied torso-ready state owns the fixed height of this plan.
    # A measured near-goal height must also reach IK, HOME and execution.
    place_height = float(torso_ready[0])
    home = HOME.copy(); home[0] = place_height
    progress('unloaded_home_preparation_begin')
    unloaded = node._plan_retracted_transition(torso_ready, home)
    if unloaded is None:
        raise RuntimeError('No safe unloaded route from carry pose to HOME')
    progress('unloaded_home_preparation_completed', unloaded_waypoints=len(unloaded))
    margin = max(node.place_joint_limit_margin, .10)
    accepted = None
    candidate_number = 0

    def candidate(solutions, unused_transition):
        nonlocal accepted, candidate_number
        candidate_number += 1
        progress('candidate_begin', candidate_number=candidate_number,
                 cartesian_waypoints=len(solutions), scene_samples=scene.samples)
        # Reject an IK branch whose Cartesian endpoints already intersect the
        # table/robot before spending time searching its carry-to-bin setup.
        if not all(scene.leg(q, q, aperture, True) for q in solutions):
            return False
        # Try fewer setup goals only for the measured bin pose. Every resulting
        # interpolation still passes the original support/body/scene checks;
        # a rejected simpler route falls through to the unchanged 20-step one.
        policies = ([dict(shoulder_progress_power=1.,
                         joint_progress_powers=(1., .5, 1., 1., .5, 1.),
                         proximal_steps=count)
                     for count in ((10, 20) if bin_scene is not None else (20,))]
                    if table is not None else [])
        policies.extend(dict(shoulder_progress_power=power) for power in (1., .5))
        for policy in policies:
            progress('candidate_setup_begin', candidate_number=candidate_number, policy=policy)
            if node._cancel.is_set():
                return False
            goals = coordinated_supported_goals(node.chain, torso_ready, solutions[0], margin=margin,
                minimum_support=node.carried_supported_jaw_vertical_component,
                **policy)
            if goals is None:
                continue
            # Cheap signed-support screening before full mesh work.
            previous = torso_ready
            for q in goals:
                if not node._gravity_supported_transition_is_safe(previous, q):
                    break
                previous = q
            else:
                route = node._plan_carried_joint_route(torso_ready, goals, attached,
                                                       require_gravity_support=True)
                if route is None:
                    continue
                progress('candidate_setup_body_completed', candidate_number=candidate_number,
                         setup_legs=len(route), scene_samples=scene.samples)
                previous = torso_ready
                for q in route:
                    if not scene.leg(previous, q, aperture, True):
                        break
                    previous = q
                else:
                    progress('candidate_cartesian_legs_begin', candidate_number=candidate_number,
                             scene_samples=scene.samples)
                    for q in solutions[1:]:
                        if (not node._carried_robot_transition_is_safe(previous, q, attached)
                                or not node._gravity_supported_transition_is_safe(previous, q)
                                or not scene.leg(previous, q, aperture, True)):
                            break
                        previous = q
                    else:
                        progress('candidate_opening_begin', candidate_number=candidate_number,
                                 scene_samples=scene.samples)
                        count = int(math.ceil(abs(open_aperture-aperture)/.001))+1
                        if not all(scene.sample(solutions[-1], float(q), True)
                                   for q in np.linspace(aperture, open_aperture, count)):
                            continue
                        # After measured release, a clear direct fold removes
                        # the unnecessary return through the loaded carry pose.
                        returns = []
                        if table is not None:
                            returns.append(([*reversed(solutions[:-1]), home, HOME], []))
                        returns.append(([*reversed(solutions[:-1]), *reversed(route[:-1]),
                                         torso_ready, *unloaded, home, HOME], None))
                        for return_goals, direct_home in returns:
                            progress('candidate_empty_return_begin', candidate_number=candidate_number,
                                     return_legs=len(return_goals), direct_home=direct_home is not None,
                                     scene_samples=scene.samples)
                            previous = solutions[-1]
                            for q in return_goals:
                                if not scene.leg(previous, q, open_aperture, False):
                                    break
                                previous = q
                            else:
                                progress('candidate_accepted', candidate_number=candidate_number,
                                         scene_samples=scene.samples)
                                accepted = (route, policy, len(return_goals), direct_home)
                                return True
        return False

    try:
        progress('cartesian_search_begin', scene_samples=scene.samples,
                 search_wall_budget_seconds=getattr(node, 'scene_cartesian_wall_seconds', 420.0))
    except Exception:
        pass
    search_diagnostics = None
    if table is not None:
        search_diagnostics = {}
        solutions, orientation, score, _ = solve_scene_cartesian(
            node, positions, rotation, place_height, staging_seed, scene, candidate,
            measured_seed=torso_ready, aperture=aperture, open_aperture=open_aperture,
            seed_templates=(HOME, OFFER, PREGRASP),
            limits=SearchLimits(wall_seconds=float(getattr(node, 'scene_cartesian_wall_seconds', 420.0))),
            diagnostics_out=search_diagnostics)
    else:
        solutions, orientation, score, _ = node._solve_cartesian_path(
            positions, (rotation,), place_height, transition_start=staging_seed,
            skip_setup_transition=True, candidate_validator=candidate, first_valid=True,
            joint_limit_margin=margin)
    progress('cartesian_search_completed', search_diagnostics=search_diagnostics,
             scene_samples=scene.samples)
    if accepted is None or node._cancel.is_set():
        raise RuntimeError('scene_checked_place_no_accepted_route')
    route, policy, return_legs, direct_home = accepted
    diagnostic = dict(
        scope='sampled_nominal_scene_checked_loaded_place', original_bin_point=obstacle.point.tolist(),
        bin_origin=obstacle.origin.tolist(), bin_rotation=obstacle.rotation.tolist(),
        bin_outer_expanded_bounds=obstacle.bounds.tolist(), modeled_margin_m=obstacle.margin,
        bin_material_solids=len(obstacle.material_bounds),
        bin_certified_cavity=(None if obstacle.cavity_bounds is None else obstacle.cavity_bounds.tolist()),
        nominal_floor_registration=bin_scene is None,
        registered_bin_scene=obstacle.registered_scene,
        whole_table_pose_modeled=table is not None,
        table_scene=table_scene,
        nominal_mimic_geometry=True, actual_passive_deflection_verified=False,
        measured_master=aperture, master_producer_stamp_ns=int(feedback.stamp_ns),
        modeled_open_master=open_aperture,
        actual_carry_start=np.asarray(carried_start).tolist(), torso_ready=np.asarray(torso_ready).tolist(),
        staging_used_as_ik_seed_only=True, old_reverse_compact_executed=False,
        cartesian_search=search_diagnostics,
        joint_limit_margin=margin, shoulder_progress_power=policy['shoulder_progress_power'],
        proximal_setup_steps=policy.get('proximal_steps', 20),
        joint_progress_powers=list(policy.get('joint_progress_powers',
            (policy['shoulder_progress_power'], 1., 1., 1., 1., 1.))),
        direct_empty_fold_from_clearance=direct_home is not None,
        scene_samples=scene.samples, exact_cache_hits=scene.cache_hits,
        setup_legs=len(route), open_return_legs=return_legs,
        open_tool_robot_self_checked=True, tool_tool_pairs_checked=False,
        head_screen_visual_box_checked=True,
        minimum_moving_left_geometry_z_m=scene.minimum_moving_left_z,
        minimum_moving_left_z_minus_bin_point_z_m=scene.minimum_moving_left_z-obstacle.point[2],
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        bin_mesh_sha256=hashlib.sha256(bin_mesh.read_bytes()).hexdigest(),
        planning_wall_seconds=time.monotonic()-started)
    return SceneCheckedPlacePlan(solutions, orientation, score, route, unloaded, diagnostic, direct_home)
