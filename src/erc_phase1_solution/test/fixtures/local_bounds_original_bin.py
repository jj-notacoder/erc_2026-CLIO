"""Exact R42 AST fixture; source SHA256 6f64dc91bd91bb0e7cef36ba3b2208f36c5caea0c16357178f758ab689214903."""
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

    def __init__(self, point, outer_bounds, *, margin=0.005, cavity_bounds=None, bin_scene=None):
        self.point = _finite(point, (3,), 'bin point')
        raw = _finite(outer_bounds, (2, 3), 'bin bounds')
        if np.any(raw[1] <= raw[0]) or not math.isfinite(margin) or margin < 0:
            raise ValueError('invalid bin extent or margin')
        self.registered_scene = None
        if bin_scene is None:
            ray = np.r_[self.point[:2], 0.0]
            if np.linalg.norm(ray) < 1e-09:
                raise ValueError('bin point has no horizontal approach ray')
            ray /= np.linalg.norm(ray)
            self.rotation = np.column_stack(([-ray[1], ray[0], 0.0], [0.0, 0.0, 1.0], ray))
            self.origin = self.point - self.rotation @ np.asarray([0.0, -0.095, 0.0])
        else:
            registered = validate_bin_scene(bin_scene)
            if not np.allclose(raw, registered['outer_bounds'], rtol=0.0, atol=1e-06) or np.linalg.norm(self.point - np.asarray(registered['floor_center'])) > 1e-06 or cavity_bounds is None or (not np.allclose(cavity_bounds, registered['cavity_bounds'], rtol=0.0, atol=1e-06)):
                raise ValueError('registered bin pose does not match the selected CAD and target')
            self.rotation = np.asarray(registered['rotation'], dtype=float)
            self.origin = np.asarray(registered['origin'], dtype=float)
            margin = max(margin, registered['modeled_margin_m']) + registered['registration_margin_m']
            self.registered_scene = registered
        self.bounds = raw + np.asarray([[-margin] * 3, [margin] * 3])
        centre = self.bounds.mean(axis=0)
        half = (self.bounds[1] - self.bounds[0]) / 2
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
        self.corners = (centre + signs * half) @ self.rotation.T + self.origin
        self.triangles = (_box_triangles(2 * half) + centre) @ self.rotation.T + self.origin
        self.margin = float(margin)
        self.material_bounds = [self.bounds]
        self.cavity_bounds = None
        if cavity_bounds is not None:
            cavity = _finite(cavity_bounds, (2, 3), 'bin cavity')
            if np.any(cavity[1] <= cavity[0]) or np.any(cavity[0] < raw[0]) or np.any(cavity[1] > raw[1] + 1e-08) or (not np.isclose(cavity[1, 1], raw[1, 1], atol=1e-08, rtol=0.0)):
                raise ValueError('cavity must be inside the bounds and open at the top')
            self.cavity_bounds = cavity
            slabs = []
            floor = raw.copy()
            floor[1, 1] = cavity[0, 1]
            slabs.append(floor)
            for axis in (0, 2):
                low = raw.copy()
                low[1, axis] = cavity[0, axis]
                slabs.append(low)
                high = raw.copy()
                high[0, axis] = cavity[1, axis]
                slabs.append(high)
            self.material_bounds = [slab + [[-margin] * 3, [margin] * 3] for slab in slabs]
        self.material_corners = []
        self.material_triangles = []
        for slab in self.material_bounds:
            centre, half = (slab.mean(axis=0), (slab[1] - slab[0]) / 2)
            self.material_corners.append((centre + signs * half) @ self.rotation.T + self.origin)
            self.material_triangles.append((_box_triangles(2 * half) + centre) @ self.rotation.T + self.origin)

    def intersects(self, surface, transform, watertight):
        """Conservative local AABB broad phase, unchanged SAT/containment narrow phase."""
        low, high = (surface.min(axis=(0, 1)), surface.max(axis=(0, 1)))
        rotation = self.rotation.T @ transform[:3, :3]
        translation = self.rotation.T @ (transform[:3, 3] - self.origin)
        centre = rotation @ ((low + high) / 2) + translation
        half = np.abs(rotation) @ ((high - low) / 2)
        if np.any(centre + half < self.bounds[0]) or np.any(centre - half > self.bounds[1]):
            return False
        world = surface @ transform[:3, :3].T + transform[:3, 3]
        return any((oriented_box_intersects_triangles(corners, world, closed_surface=watertight) for corners, slab in zip(self.material_corners, self.material_bounds) if not (np.any(centre + half < slab[0]) or np.any(centre - half > slab[1]))))

    def book_intersects(self, corners):
        return any((oriented_box_intersects_triangles(corners, surface, closed_surface=True) for surface in self.material_triangles))
