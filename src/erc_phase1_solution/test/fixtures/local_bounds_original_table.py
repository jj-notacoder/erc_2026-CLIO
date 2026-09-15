"""Exact R42 AST fixture; source SHA256 f98f10eea9b527ac1e207dab6157e3fdb43af297fec4e2e227dab5197d42e991."""
def _finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError('invalid ' + name)
    return result.copy()

def _rigid(value):
    t = _finite(value, (4, 4), 'camera transform')
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-09) or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-06) or abs(np.linalg.det(t[:3, :3]) - 1) > 1e-06:
        raise ValueError('nonrigid camera transform')
    return t

class TableSceneObstacle:
    """Five finite expanded solids; compatible with PlaceSceneChecker.intersects."""

    def __init__(self, scene):
        if not isinstance(scene, dict) or scene.get('valid') is not True:
            raise ValueError('table scene is not registered')
        if scene.get('model') != 'erc_table_two_edge_rgbd_v1' or scene.get('frame') != 'base_footprint':
            raise ValueError('unknown table scene/frame')
        self.origin = _finite(scene['origin'], (3,), 'table origin')
        transform = np.eye(4)
        transform[:3, :3] = _finite(scene['rotation'], (3, 3), 'table rotation')
        _rigid(transform)
        self.rotation = transform[:3, :3]
        modeled = float(scene['modeled_margin_m'])
        registration = float(scene['registration_margin_m'])
        margin = modeled + registration
        if not math.isfinite(modeled) or not math.isfinite(registration) or modeled < MODELED_MARGIN or (registration < 0) or (not MODELED_MARGIN <= margin <= 0.12):
            raise ValueError('invalid table margin')
        solids = scene.get('solids', [])
        if solids != _solid_bounds() or scene.get('mesh_sha256') != TABLE_MESH_SHA256:
            raise ValueError('table CAD bounds/hash mismatch')
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
        self.solids = []
        self.solid_triangles = []
        faces = np.asarray([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        for solid in solids:
            bounds = np.asarray(solid['bounds']) + np.array([[-margin] * 3, [margin] * 3])
            centre, half = (bounds.mean(axis=0), (bounds[1] - bounds[0]) / 2)
            corners = (centre + signs * half) @ self.rotation.T + self.origin
            self.solids.append((solid['name'], bounds, corners))
            self.solid_triangles.append(corners[faces])
        self.margin = margin
        self.last_intersection = None

    def intersects(self, surface, transform, watertight):
        from erc_phase1_solution.kinematics import oriented_box_intersects_triangles
        surface = np.asarray(surface, dtype=float)
        if surface.ndim != 3 or surface.shape[1:] != (3, 3) or (not len(surface)) or (not np.all(np.isfinite(surface))):
            raise ValueError('invalid table collision surface')
        transform = _rigid(transform)
        lo, hi = (surface.min(axis=(0, 1)), surface.max(axis=(0, 1)))
        rotation = self.rotation.T @ transform[:3, :3]
        translation = self.rotation.T @ (transform[:3, 3] - self.origin)
        centre = rotation @ ((lo + hi) / 2) + translation
        half = np.abs(rotation) @ ((hi - lo) / 2)
        world = None
        self.last_intersection = None
        for name, bounds, corners in self.solids:
            if np.any(centre + half < bounds[0]) or np.any(centre - half > bounds[1]):
                continue
            if world is None:
                world = surface @ transform[:3, :3].T + transform[:3, 3]
            if oriented_box_intersects_triangles(corners, world, closed_surface=bool(watertight)):
                self.last_intersection = name
                return True
        return False

    def intersects_box(self, corners):
        """Check held book OBB against each table solid, including containment."""
        from erc_phase1_solution.kinematics import oriented_box_intersects_triangles
        corners = _finite(corners, (8, 3), 'held book corners')
        self.last_intersection = None
        for (name, _, _), triangles in zip(self.solids, self.solid_triangles):
            if oriented_box_intersects_triangles(corners, triangles, closed_surface=True):
                self.last_intersection = name
                return True
        return False
