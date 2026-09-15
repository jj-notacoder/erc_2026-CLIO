"""Attempt-local, sampled empty PICK robot/tool checks; no physics changes.

Reuses the existing robot collision predicate and nominal official gripper
linkage. The screen's URDF visual box is an additional planner obstacle only.
This does not change Gazebo's collision model or prove passive deflection bounds.
"""
from __future__ import annotations

import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import (
    PreparedTriangleMesh, URDFChain, _numbers, _rpy_matrix,
    oriented_box_intersects_triangles, triangle_meshes_intersect,
)
from .motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS, PREGRASP
from .rigid_palm_preflight import LEFT_GRIPPER_COLLISION_LINKS, PALM_COLLISION_LINK
from .shelf_cradle_geometry import ShelfCradleGeometry
from .static_pair_separation import separated_on_axes

HEAD = ('head_1_joint', 'head_2_joint')
MASTER = 'gripper_left_finger_joint'
NAMES = (*IK_JOINTS, *RIGHT_ARM_JOINTS, *HEAD, MASTER)


def measured_context(node):
    """Use the existing empty-preparation 350 ms joint freshness allowance."""
    with node._lock:
        joints = dict(node.joints)
        stamps = dict(node._joint_stamps_ns)
    now = node.get_clock().now().nanoseconds
    for name in NAMES:
        if (name not in joints or not math.isfinite(joints[name])
                or not -.05e9 <= now - stamps.get(name, 0) <= .35e9):
            raise RuntimeError(f'empty pickup joint measurement stale: {name}')
    return joints, {name: int(stamps[name]) for name in NAMES}


def _fixed(values, size):
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError('empty pickup context must contain finite joint values')
    return np.frombuffer(result.tobytes(), dtype=float)


def _bounds(surface):
    return np.array([surface.min(axis=(0, 1)), surface.max(axis=(0, 1))])


def joint_velocity_limits(urdf):
    """Read the deployed official limits; never infer actuator speed from a goal."""
    root = ET.parse(urdf).getroot()
    values = []
    for name in IK_JOINTS:
        joint = root.find(f"joint[@name='{name}']")
        limit = None if joint is None else joint.find('limit')
        velocity = math.nan if limit is None else float(limit.get('velocity', 'nan'))
        if not math.isfinite(velocity) or velocity <= 0:
            raise ValueError(f'empty pickup joint velocity limit invalid: {name}')
        values.append(velocity)
    return _fixed(values, len(IK_JOINTS))


class ScreenEnvelope:
    """Exact visual-box extent; surrounding head collision meshes stay active."""
    def __init__(self, urdf):
        root = ET.parse(urdf).getroot()
        visuals = root.findall("link[@name='head_screen_link']/visual")
        if len(visuals) != 1 or visuals[0].find('geometry/box') is None:
            raise ValueError('unsupported head screen visual geometry')
        visual = visuals[0]
        half = .5 * _numbers(visual.find('geometry/box').get('size'), (0, 0, 0))
        if not np.all(np.isfinite(half)) or np.any(half <= 0):
            raise ValueError('invalid head screen visual extent')
        origin = visual.find('origin')
        xyz = _numbers(None if origin is None else origin.get('xyz'), (0, 0, 0))
        rotation = _rpy_matrix(_numbers(None if origin is None else origin.get('rpy'), (0, 0, 0)))
        signs = np.array([(x, y, z) for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
        self.corners = signs * half @ rotation.T + xyz
        self.chain = URDFChain.from_urdf(urdf, 'base_footprint', 'head_screen_link',
                                      ('torso_lift_joint', *HEAD))

    def world(self, torso, head):
        transform = self.chain.forward([torso, *head])
        return self.corners @ transform[:3, :3].T + transform[:3, 3]


class EmptyPickupCollision:
    """One immutable sensor context and exact sample cache for one PICK attempt."""
    def __init__(self, node, left, right, head, aperture, *, geometry, screen):
        self.node = node
        self.start = _fixed(left, len(IK_JOINTS))
        self.right = _fixed(right, len(RIGHT_ARM_JOINTS))
        self.head = _fixed(head, 2)
        self.initial_aperture = float(aperture)
        self.open_aperture = float(node.gripper_open)
        if (not math.isfinite(self.initial_aperture) or not math.isfinite(self.open_aperture)
                or not 0 <= self.open_aperture <= .069
                or not -1e-6 <= self.initial_aperture <= .069 + 1e-6):
            raise ValueError('empty pickup aperture outside measured/command bounds')
        self.geometry, self.screen = geometry, screen
        self._world_geometry_cache = None
        if (type(self) is EmptyPickupCollision and type(geometry) is ShelfCradleGeometry
                and type(screen) is ScreenEnvelope):
            from .exact_world_geometry import ExactWorldGeometry, ordinary_model_producer
            if ordinary_model_producer(node):
                self._world_geometry_cache = ExactWorldGeometry()
        self.context = dict(right_positions=self.right, head_positions=self.head)
        self.cache = {}
        self.checked_samples = self.cache_hits = 0
        self.last_rejection = None
        self.last_rejected_q = None
        self.last_rejected_aperture = None

    @classmethod
    def capture(cls, node):
        joints, stamps = measured_context(node)
        from ament_index_python.packages import get_package_share_directory
        urdf = Path(get_package_share_directory('erc_description')) / 'urdf/tiago_pro.urdf'
        geometry = getattr(node, '_shelf_cradle_geometry', None)
        if geometry is None:
            geometry = ShelfCradleGeometry(urdf, get_package_share_directory, immutable_local=True)
            node._shelf_cradle_geometry = geometry
        result = cls(node, [joints[n] for n in IK_JOINTS], [joints[n] for n in RIGHT_ARM_JOINTS],
                     [joints[n] for n in HEAD], joints[MASTER], geometry=geometry, screen=ScreenEnvelope(urdf))
        result.stamps = stamps
        result.velocity_limits = joint_velocity_limits(urdf)
        return result

    def _reject(self, reason):
        self.last_rejection = reason
        return reason

    def sample(self, q, aperture):
        if self.node._cancel.is_set():
            return self._reject('empty_pickup_cancelled')
        q = _fixed(q, len(IK_JOINTS))
        aperture = float(aperture)
        if (not math.isfinite(aperture) or not 0 <= aperture <= .069
                or np.any(q < self.node.chain.lower) or np.any(q > self.node.chain.upper)):
            return self._reject('empty_pickup_joint_limit')
        key = (q.tobytes(), np.float64(aperture).tobytes())
        if key in self.cache:
            self.cache_hits += 1
            reason = self.cache[key]
            if reason:
                self.last_rejected_q = q.tolist()
                self.last_rejected_aperture = aperture
                return self._reject(reason)
            return None
        self.checked_samples += 1
        reason = self._sample_uncached(q, aperture)
        self.cache[key] = reason
        if reason:
            self.last_rejected_q = q.tolist()
            self.last_rejected_aperture = aperture
        return self._reject(reason) if reason else None

    def _cached_robot_world(self, q):
        """Reuse immutable mesh/transform bytes; preserve component traversal."""
        transforms = self.node._collision_link_transforms(q, **self.context)
        grouped, single_geometry = {}, {}
        for mesh in self.node.carried_collision_meshes:
            transform = transforms[mesh.link]
            cached = self._world_geometry_cache.capture(
                mesh.local_mesh, mesh.triangles, transform)
            surface = (mesh.triangles @ transform[:3, :3].T + transform[:3, 3]
                       if cached is None else cached.views()[0])
            grouped.setdefault(mesh.link, []).append(surface)
            if len(grouped[mesh.link]) == 1 and cached is not None:
                single_geometry[mesh.link] = cached
            else:
                single_geometry.pop(mesh.link, None)
        robot = {link: values[0] if len(values) == 1 else np.concatenate(values)
                 for link, values in grouped.items()}
        return robot, single_geometry

    def _sample_uncached(self, q, aperture):
        # The ordinary body and tool checks use the same exact world facets.
        # Custom producers keep the original body-first call path.
        from erc_phase1_solution.exact_world_geometry import ordinary_model_producer
        from erc_phase1_solution.sample_collision_snapshot import _capture_scene_robot_snapshot
        robot = snapshot = None
        if (ordinary_model_producer(self.node)
                and type(self.geometry) is ShelfCradleGeometry
                and type(self.screen) is ScreenEnvelope):
            geometry_options = {}
            if self._world_geometry_cache is None:
                robot = self.node._world_collision_surfaces(q, **self.context)
            else:
                robot, cached_geometry = self._cached_robot_world(q)
                geometry_options['_world_geometry'] = cached_geometry
            snapshot = _capture_scene_robot_snapshot(
                self.node, q, self.right, self.head, robot, **geometry_options)
        collision = (self.node._robot_self_collision(q, **self.context)
                     if snapshot is None else self.node._robot_self_collision(
                         q, **self.context, _scene_snapshot=snapshot))
        if collision is not None:
            return f'empty_pickup_robot_self:{collision}'
        if robot is None:
            robot = self.node._world_collision_surfaces(q, **self.context)
        # A body-cache hit still needs robot bounds below. Resolve the already
        # captured immutable world so cached extrema are reused in that case.
        resolver = (snapshot.resolve if self._world_geometry_cache is not None
                    else snapshot.resolved) if snapshot is not None else None
        shared = (None if resolver is None else resolver(
            self.node, q, self.right, self.head))
        if shared is not None:
            robot = shared[0]
        closed = self.node._watertight_collision_links()
        local = self.geometry.local_surfaces(aperture)
        if set(local) != set(LEFT_GRIPPER_COLLISION_LINKS):
            return 'empty_pickup_tool_inventory_invalid'
        hand = self.node.chain.forward(q)
        tool = {link: triangles @ hand[:3, :3].T + hand[:3, 3] for link, triangles in local.items()}
        screen = self.screen.world(q[0], self.head)
        screen_bounds = np.array([screen.min(axis=0), screen.max(axis=0)])
        bounds = {link: (_bounds(surface) if shared is None or link in tool
                         else shared[1][link])
                  for link, surface in {**robot, **tool}.items()}
        prepared = {}
        for link in robot:
            if link.startswith('arm_left_') and bounds[link][0, 2] < .02:
                return f'empty_pickup_robot_floor:{link}'
        for link, surface in {**robot, **tool}.items():
            # Screen is rigidly part of the head assembly; test against robot
            # body/arms and complete left tool, not its own surrounding cover.
            if link in ('head_1_link', 'head_2_link', 'head_front_camera_link'):
                continue
            b = bounds[link]
            if np.any(b[1] < screen_bounds[0]) or np.any(screen_bounds[1] < b[0]):
                continue
            if oriented_box_intersects_triangles(screen, surface,
                    closed_surface=link in closed or self.geometry.watertight.get(link, False)):
                return f'empty_pickup_screen:{link}'
        for tool_link, surface in tool.items():
            b = bounds[tool_link]
            if b[0, 2] < .02:
                return f'empty_pickup_tool_floor:{tool_link}'
            for robot_link, other in robot.items():
                if tool_link == PALM_COLLISION_LINK and robot_link == 'arm_left_7_link':
                    continue  # Same adjacent wrist/palm exclusion as existing tool sweep.
                ob = bounds[robot_link]
                if np.any(b[1] < ob[0]) or np.any(ob[1] < b[0]):
                    continue
                # Same current-vertex proof as PLACE; uncertain pairs retain
                # the original detailed predicate and containment checks.
                if separated_on_axes(surface, other, hand[:3, :3].T):
                    continue
                for link, triangles in ((tool_link, surface), (robot_link, other)):
                    if link not in prepared:
                        prepared[link] = PreparedTriangleMesh(triangles)
                if triangle_meshes_intersect(prepared[tool_link], prepared[robot_link],
                        first_watertight=self.geometry.watertight[tool_link], second_watertight=robot_link in closed):
                    return f'empty_pickup_tool_robot:{tool_link}:{robot_link}'
        return None

    def edge(self, first, last, aperture=None):
        """Controller position-only joint segment; same <=.02/61 sample policy."""
        first, last = _fixed(first, len(IK_JOINTS)), _fixed(last, len(IK_JOINTS))
        aperture = self.open_aperture if aperture is None else aperture
        count = max(61, int(self.node.carried_transition_samples),
                    int(math.ceil(float(np.max(np.abs(last - first))) / .02)) + 1)
        if np.array_equal(first, last):
            count = 1
        sequence = getattr(self, '_parallel_sequence', None)
        if sequence is not None and math.isfinite(float(aperture)):
            return sequence((first + fraction * (last - first), float(aperture))
                            for fraction in np.linspace(0., 1., count))
        for fraction in np.linspace(0., 1., count):
            if self.sample(first + fraction * (last - first), aperture):
                return False
        return True

    def opening(self):
        # Numerical measured-bound residual is preserved in diagnostics; only
        # nominal opening geometry is clipped to the legal command range.
        first = float(np.clip(self.initial_aperture, 0., .069))
        count = max(1, int(math.ceil(abs(self.open_aperture - first) / .001)) + 1)
        sequence = getattr(self, '_parallel_sequence', None)
        if sequence is not None:
            return sequence((self.start, float(a)) for a in np.linspace(first, self.open_aperture, count))
        return all(self.sample(self.start, a) is None for a in np.linspace(first, self.open_aperture, count))

    def retracted_edge(self, first, last):
        # Existing search rechecks the same cheap bound. Reject it here before
        # expensive meshes; this changes no accepted edge or global predicate.
        return self.node._retracted_transition_is_safe(first, last) and self.edge(first, last)

    def plan_transition(self, first, last):
        """Bounded empty setup search; every returned controller leg is checked.

        Low-wrist waypoints turn the open fingers below the head before the
        shoulder completes its lift. These are legal-joint planning templates,
        not measured poses or collision exemptions. The exact IK goal remains.
        """
        first, last = _fixed(first, len(IK_JOINTS)), _fixed(last, len(IK_JOINTS))
        if self.node._cancel.is_set():
            return None
        if self.retracted_edge(first, last):
            return []
        pregrasp = np.asarray(PREGRASP, dtype=float).copy()
        pregrasp[0] = last[0]
        candidates = [pregrasp]
        for shoulder in (.5, .3, .7):
            for elbow in (-2.1, -2.25):
                waypoint = last.copy()
                waypoint[2], waypoint[4] = shoulder, elbow
                waypoint[6:] = first[6:]
                candidates.append(waypoint)
        for waypoint in candidates:
            if self.node._cancel.is_set():
                return None
            # Reject cheap workspace/limit failures before expensive meshes.
            if (np.any(waypoint < self.node.chain.lower)
                    or np.any(waypoint > self.node.chain.upper)
                    or not self.node._retracted_transition_is_safe(first, waypoint)
                    or not self.node._retracted_transition_is_safe(waypoint, last)):
                continue
            if self.sample(waypoint, self.open_aperture) is not None:
                continue
            if self.edge(first, waypoint) and self.edge(waypoint, last):
                if self.node._cancel.is_set():
                    return None
                return [waypoint]
        return None

    def candidate(self, solutions, transition):
        elevated = self.start.copy()
        elevated[0] = self.node.pick_torso_height
        previous = elevated
        for goal in (*transition, *solutions):
            if not self.edge(previous, goal):
                return False
            previous = goal
        return True

    def require_fresh(self, expected, aperture, *, return_context=False):
        """Existing empty-setup engineering drift gates, not displacement proof."""
        if self.node._cancel.is_set():
            raise RuntimeError('empty pickup cancelled before command')
        current, stamps = measured_context(self.node)
        if abs(current['torso_lift_joint'] - float(expected[0])) > .002:
            raise RuntimeError('empty pickup torso differs from checked geometry')
        for names, values, tolerance in ((IK_JOINTS, expected, .008),
                                        (RIGHT_ARM_JOINTS, self.right, .003), (HEAD, self.head, .003)):
            if any(abs(current[n] - float(v)) > tolerance for n, v in zip(names, values)):
                raise RuntimeError('empty pickup collision context changed before command')
        if abs(current[MASTER] - aperture) > float(self.node.adaptive_endpoint_tolerance):
            raise RuntimeError('empty pickup gripper differs from checked geometry')
        if return_context:
            return current, stamps

    def wait_for_endpoint(self, previous, expected, *, aperture, phase):
        """Wait after a successful empty-PICK action, with the same context gates."""
        return wait_for_geometry_endpoint(self.node, previous, expected,
            right=self.right, head=self.head, velocity_limits=self.velocity_limits,
            aperture=aperture, phase=phase, command='pick')


def wait_for_geometry_endpoint(node, previous, expected, *, right, head,
                               velocity_limits, aperture, phase, command,
                               context_check=None):
    """Bounded fresh endpoint observation; no action or publisher is used.

    Moving joints may converge within the existing 2mm/8mrad geometry bounds.
    Parked joints, right arm, head and gripper remain checked every iteration.
    PLACE additionally supplies its existing live scene/contact checks.
    """
    if command not in ('pick', 'place'):
        raise ValueError('unsupported empty endpoint command')
    event_prefix = 'empty_pickup_endpoint' if command == 'pick' else 'empty_return_endpoint'
    right = _fixed(right, len(RIGHT_ARM_JOINTS))
    head = _fixed(head, len(HEAD))
    previous = _fixed(previous, len(IK_JOINTS))
    target = _fixed(expected, len(IK_JOINTS))
    limits = _fixed(velocity_limits, len(IK_JOINTS))
    if np.any(limits <= 0):
        raise ValueError('empty geometry endpoint requires positive velocity limits')
    moving = np.abs(target - previous) > 1e-12
    tolerances = np.asarray([.002, *([.008] * (len(IK_JOINTS)-1))])
    minimum_travel_seconds = float(np.max(np.abs(target - previous) / limits))
    # Allow the whole nominal travel time after controller success, plus
    # two seconds. This is a bounded engineering wait, not a speed promise.
    ros_budget = minimum_travel_seconds + 2.0
    wall_budget = min(float(node.timeout), max(5.0, 4.0 * ros_budget + 2.0))
    if not math.isfinite(wall_budget) or wall_budget <= 0:
        raise ValueError('empty geometry endpoint wall budget is invalid')
    started_ns = int(node.get_clock().now().nanoseconds)
    last_seen_ns = started_ns
    started_wall = time.monotonic()
    last = None
    waiting_reported = False

    def report(event, reason=None):
        node._publish_status(event, command=command, phase=phase,
            reason=reason, previous=previous.tolist(), expected=target.tolist(),
            joint_tolerances=tolerances.tolist(), velocity_limits=limits.tolist(),
            minimum_nominal_travel_seconds=minimum_travel_seconds,
            action_return_ros_ns=started_ns, ros_wait_budget_seconds=ros_budget,
            wall_wait_budget_seconds=wall_budget,
            elapsed_wall_seconds=time.monotonic()-started_wall,
            last_evaluated_measurement=last,
            scope='Fresh measured endpoint within unchanged geometry tolerances; no new command or geometry recomputation')

    def fail(reason):
        report(event_prefix + '_rejected', reason)
        raise RuntimeError(reason)

    while True:
        now_ns = int(node.get_clock().now().nanoseconds)
        if node._cancel.is_set():
            fail('empty geometry cancelled during endpoint wait')
        if now_ns < last_seen_ns:
            fail('empty geometry clock reversed during endpoint wait')
        last_seen_ns = now_ns
        if (now_ns-started_ns >= int(ros_budget*1e9)
                or time.monotonic()-started_wall >= wall_budget):
            fail('empty geometry measured endpoint timed out')
        try:
            if context_check is not None:
                context_check()
            current, stamps = measured_context(node)
        except RuntimeError as error:
            fail(str(error))
        measured = np.asarray([current[n] for n in IK_JOINTS])
        errors = np.abs(measured-target)
        last = dict(evaluated_ros_ns=now_ns, measured_left=measured.tolist(),
                    absolute_joint_errors=errors.tolist(), producer_stamps_ns=stamps,
                    measured_right=[current[n] for n in RIGHT_ARM_JOINTS],
                    measured_head=[current[n] for n in HEAD], measured_aperture=current[MASTER])
        for names, values in ((RIGHT_ARM_JOINTS, right), (HEAD, head)):
            if any(abs(current[n]-float(v)) > .003 for n, v in zip(names, values)):
                fail('empty geometry parked context changed during endpoint wait')
        if abs(current[MASTER]-aperture) > float(node.adaptive_endpoint_tolerance):
            fail('empty geometry gripper differs during endpoint wait')
        if np.any(errors[~moving] > tolerances[~moving]):
            fail('empty geometry uncommanded joint changed during endpoint wait')
        # Require a producer sample newer than action return, in addition
        # to measured_context's original 350 ms freshness/future policy.
        after_action = all(stamps[n] > started_ns for n in IK_JOINTS)
        if np.all(errors <= tolerances) and after_action:
            if node._cancel.is_set():
                fail('empty geometry cancelled during endpoint wait')
            if context_check is not None:
                try:
                    context_check()
                except RuntimeError as error:
                    fail(str(error))
            report(event_prefix + '_verified')
            return measured
        if not waiting_reported:
            report(event_prefix + '_waiting')
            waiting_reported = True
        time.sleep(.02)
