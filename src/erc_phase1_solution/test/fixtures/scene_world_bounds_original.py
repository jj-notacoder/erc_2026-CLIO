"""Original scene checker for bounds-cache regression comparisons.
Extracted from certified final08 scene_checked_place.py SHA-256
06b57f61f2257fb579c1f0df271a1bb1a703e419c53ec12af5c7c947a86b7438.
Only the named definitions are compiled by the tests.
"""

def _finite(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f'invalid {label}')
    return result.copy()

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
        robot = {}
        for mesh in node.carried_collision_meshes:
            transform = transforms[mesh.link]
            if self.obstacle.intersects(mesh.triangles, transform, mesh.watertight):
                return self._reject('robot_bin', link=mesh.link)
            if self.table is not None and self.table.intersects(mesh.triangles, transform, mesh.watertight):
                return self._reject('robot_table', link=mesh.link, solid=self.table.last_intersection)
            world = mesh.triangles @ transform[:3, :3].T + transform[:3, 3]
            robot.setdefault(mesh.link, []).append(world)
        robot = {link: values[0] if len(values) == 1 else np.concatenate(values)
                 for link, values in robot.items()}
        hand = node.chain.forward(q)
        tool = {}
        for link, surface in self.tool.local_surfaces(aperture).items():
            if self.obstacle.intersects(surface, hand, self.tool.watertight[link]):
                return self._reject('tool_bin', link=link)
            if self.table is not None and self.table.intersects(surface, hand, self.tool.watertight[link]):
                return self._reject('tool_table', link=link, solid=self.table.last_intersection)
            tool[link] = surface @ hand[:3, :3].T + hand[:3, 3]
        if self.screen is not None:
            screen = self.screen.world(q[0], head)
            low, high = screen.min(axis=0), screen.max(axis=0)
            closed = node._watertight_collision_links()
            for link, surface in {**robot, **tool}.items():
                if link in ('head_1_link', 'head_2_link', 'head_front_camera_link'):
                    continue
                if np.any(surface.max(axis=(0, 1)) < low) or np.any(high < surface.min(axis=(0, 1))):
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
        bounds = {link: np.asarray([surface.min(axis=(0, 1)), surface.max(axis=(0, 1))])
                  for link, surface in robot.items()}
        closed = node._watertight_collision_links()
        prepared_robot, prepared_tool = {}, {}
        for link, bound in bounds.items():
            if link.startswith('arm_left_'):
                self.minimum_moving_left_z = min(self.minimum_moving_left_z, float(bound[0, 2]))
                if bound[0, 2] < .02:
                    return self._reject('left_arm_ground', link=link)
        for link, surface in tool.items():
            low, high = surface.min(axis=(0, 1)), surface.max(axis=(0, 1))
            self.minimum_moving_left_z = min(self.minimum_moving_left_z, float(low[2]))
            if low[2] < .02:
                return self._reject('tool_ground', link=link)
            for other, other_surface in robot.items():
                if link == PALM_COLLISION_LINK and other == 'arm_left_7_link':
                    continue
                other_low, other_high = bounds[other]
                if np.any(high < other_low) or np.any(other_high < low):
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
