"""Exact R51 source extracts; original full module SHA 8ab6dcfdf2e11d0d9e0f38aae1f8858c3e9eeb3728e29375be70dbabb4dffe84."""
from __future__ import annotations

def _finite(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f'invalid {label}')
    return result.copy()

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
        snapshot = None
        if (type(self.obstacle) is NominalBinObstacle
                and (self.table is None or type(self.table) is TableSceneObstacle)
                and (self.screen is None or type(self.screen) is ScreenEnvelope)
                and type(self.tool) is ShelfCradleGeometry):
            from erc_phase1_solution.sample_collision_snapshot import _capture_scene_robot_snapshot
            snapshot = _capture_scene_robot_snapshot(node, q, right, head, robot)
        collision = (node._robot_self_collision(q, **context) if snapshot is None
                     else node._robot_self_collision(q, **context, _scene_snapshot=snapshot))
        if collision is not None:
            return self._reject('robot_self', pair=list(collision))
        shared_geometry = (snapshot.resolved(node, q, right, head)
                           if snapshot is not None else None)
        if shared_geometry is not None:
            robot, bounds = shared_geometry
        else:
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
