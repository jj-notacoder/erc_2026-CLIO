"""Finite stock-table registration from onboard aligned RGB-D; no ROS or truth poses.

The bin's observed interior floor only selects a nearby horizontal plane. Two
exposed, perpendicular table edges locate the finite CAD rectangle. Hidden
edges and legs use the known CAD shape, never a centered-bin/world-pose prior.
Fit residuals are not calibration error bounds. The returned engineering
allowance also covers sampled pixel-edge location and a stated angular bound.
"""
from __future__ import annotations

import itertools
import math
import numpy as np

from .exact_local_bounds import ExactLocalBounds


TABLE_MESH_SHA256 = '9a662650c305d09e67d112b1c3311c78c8b75454eb05c8c7981292eb1d354977'
TABLE_LONG = 1.4
TABLE_SHORT = .8
TABLE_THICKNESS = .04
TABLE_HEIGHT = .73
LEG_WIDTH = .07
BIN_FLOOR_ABOVE_BOTTOM = .010
MODELED_MARGIN = .005


def _finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError('invalid ' + name)
    return result.copy()


def _rigid(value):
    t = _finite(value, (4, 4), 'camera transform')
    if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            or abs(np.linalg.det(t[:3, :3])-1) > 1e-6):
        raise ValueError('nonrigid camera transform')
    return t


def _invalid(reason, **quality):
    return dict(valid=False, model='erc_table_two_edge_rgbd_v1', reason=reason,
                frame='base_footprint', quality=quality)


def _fit_line(points, tolerance=.006, perpendicular_to=None):
    """Bounded deterministic RANSAC, then orthogonal least squares."""
    if len(points) < 20:
        return None
    rng = np.random.default_rng(731)
    best = None
    for _ in range(320):
        a, b = points[rng.choice(len(points), 2, replace=False)]
        delta = b-a
        length = np.linalg.norm(delta)
        if length < .10:
            continue
        if perpendicular_to is not None and abs(float((delta/length) @ perpendicular_to)) > math.sin(math.radians(5)):
            continue
        normal = np.array([-delta[1], delta[0]]) / length
        mask = np.abs((points-a) @ normal) <= tolerance
        count = int(mask.sum())
        if best is None or count > best[0]:
            best = count, mask
    if best is None or best[0] < 20:
        return None
    mask = best[1]
    for _ in range(2):
        p = points[mask]
        centre = p.mean(axis=0)
        _, _, vh = np.linalg.svd(p-centre, full_matrices=False)
        direction, normal = vh[0], vh[1]
        mask = np.abs((points-centre) @ normal) <= tolerance
    p = points[mask]
    along = (p-centre) @ direction
    span = float(np.quantile(along, .98)-np.quantile(along, .02))
    rms = float(np.sqrt(np.mean(((p-centre) @ normal)**2)))
    return dict(points=p, mask=mask, centre=centre, direction=direction,
                span=span, rms=rms)


def _solid_bounds():
    solids = [dict(name='tabletop', bounds=[[-.7, -.4, -.04], [.7, .4, 0.]])]
    for sx, sy in itertools.product((-1, 1), repeat=2):
        x = sorted([sx*.63, sx*.7]); y = sorted([sy*.33, sy*.4])
        solids.append(dict(name=f'leg_{sx}_{sy}',
                           bounds=[[x[0], y[0], -.73], [x[1], y[1], -.04]]))
    return solids


def fit_table_scene(rgb, depth, intrinsics, camera_to_base, bin_floor_point):
    """Return JSON-safe registration/quality, or valid=False with a concrete reason.

    depth is aligned optical Z in metres, intrinsics is 3x3 or its nine values,
    camera_to_base is the rigid transform at the depth producer epoch, and the
    bin point is an independently verified onboard floor point in that frame.
    RGB is used only to verify aligned image dimensions; color does not turn an
    occluding hand/book edge into a table boundary. Caller owns stamp/identity,
    camera calibration, base-frame rebasing, and scene invalidation after impact.
    """
    d = np.asarray(depth, dtype=float)
    rgb = np.asarray(rgb)
    if d.ndim != 2 or min(d.shape) < 16 or rgb.shape != d.shape+(3,):
        raise ValueError('unaligned RGB/depth images')
    k = np.asarray(intrinsics, dtype=float)
    if k.size != 9:
        raise ValueError('invalid intrinsics')
    k = _finite(k.reshape(3, 3), (3, 3), 'intrinsics')
    if (k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])
            or k[0, 1] != 0 or k[1, 0] != 0):
        raise ValueError('invalid pinhole intrinsics')
    t = _rigid(camera_to_base)
    floor = _finite(bin_floor_point, (3,), 'bin floor point')
    valid = np.isfinite(d) & (d > .15) & (d < 4.)
    safe = np.where(valid, d, 0.)
    v, u = np.indices(d.shape)
    camera = np.stack(((u-k[0, 2])*safe/k[0, 0],
                       (v-k[1, 2])*safe/k[1, 1], safe), axis=-1)
    cloud = camera @ t[:3, :3].T + t[:3, 3]
    near = np.linalg.norm(cloud[:, :, :2]-floor[:2], axis=2) <= 1.65
    expected_z = floor[2]-BIN_FLOOR_ABOVE_BOTTOM
    mask = valid & near & (abs(cloud[:, :, 2]-expected_z) <= .006)
    if mask.sum() < 300:
        return _invalid('insufficient_tabletop_plane', plane_points=int(mask.sum()))
    # Trim unrelated equal-height wall/robot pixels; retain the planar table.
    p = cloud[mask]
    for _ in range(3):
        centre = p.mean(axis=0)
        _, singular, vh = np.linalg.svd(p-centre, full_matrices=False)
        normal = vh[-1] * (1 if vh[-1, 2] >= 0 else -1)
        h = float(centre @ normal)
        p = p[abs(p @ normal-h) <= .0015]
        if len(p) < 300:
            return _invalid('insufficient_planar_support', plane_points=len(p))
    if normal[2] < math.cos(math.radians(5)) or singular[1]/math.sqrt(len(p)) < .08:
        return _invalid('tabletop_plane_not_horizontal_or_spread')
    if abs(float(floor @ normal-h)-BIN_FLOOR_ABOVE_BOTTOM) > .006:
        return _invalid('bin_floor_table_height_mismatch')
    plane_rms = float(np.sqrt(np.mean((p @ normal-h)**2)))
    if plane_rms > .001:
        return _invalid('tabletop_plane_residual', plane_rms_m=plane_rms)
    mask = valid & near & (abs(cloud @ normal-h) <= .002)
    edge = np.zeros(d.shape, dtype=bool)
    lower = valid & ((cloud @ normal) < h-.05)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        edge |= mask & np.roll(lower, (dy, dx), axis=(0, 1))
    edge[:2] = False; edge[-2:] = False; edge[:, :2] = False; edge[:, -2:] = False
    ep = cloud[edge]
    if len(ep) < 50:
        return _invalid('insufficient_exposed_table_edges', edge_points=len(ep))
    basis_x = np.array([1., 0., 0.])-normal*normal[0]
    basis_x /= np.linalg.norm(basis_x)
    basis = np.column_stack((basis_x, np.cross(normal, basis_x)))
    ep2 = ep @ basis
    first = _fit_line(ep2)
    second = None if first is None else _fit_line(ep2[~first['mask']], perpendicular_to=first['direction'])
    if first is None or second is None or min(first['span'], second['span']) < .15:
        return _invalid('two_exposed_table_edges_not_identifiable', edge_points=len(ep))
    dot = abs(float(first['direction'] @ second['direction']))
    if dot > math.sin(math.radians(3)):
        return _invalid('table_edges_not_perpendicular', absolute_edge_dot=dot)
    # Long/short assignment must be observable from the exposed span/plane.
    # A short glimpse of two edges does not justify guessing the hidden size.
    if first['span'] < second['span']:
        first, second = second, first
    if first['span'] <= TABLE_SHORT+.02 or first['span'] > TABLE_LONG+.02:
        return _invalid('ambiguous_table_dimensions', edge_spans_m=[first['span'], second['span']])
    along = first['direction'].copy()
    cross = np.array([-along[1], along[0]])
    # Orthogonalized far/side lines, located by medians of their own evidence.
    long_edge_level = float(np.median(first['points'] @ cross))
    short_edge_level = float(np.median(second['points'] @ along))
    corner = cross*long_edge_level + along*short_edge_level
    floor2 = floor @ basis
    sign_long = 1 if (floor2-corner) @ along >= 0 else -1
    sign_short = 1 if (floor2-corner) @ cross >= 0 else -1
    centre2 = corner + sign_long*along*(TABLE_LONG/2) + sign_short*cross*(TABLE_SHORT/2)
    rotation = np.column_stack((basis @ along, basis @ cross, normal))
    origin = basis @ centre2 + normal*h
    local = (cloud[mask]-origin) @ rotation
    outside = np.any(abs(local[:, :2]) > np.array([.7, .4])+.012, axis=1)
    if float(np.mean(outside)) > .02:
        return _invalid('table_rectangle_does_not_cover_plane', outside_fraction=float(np.mean(outside)))
    floor_local = (floor-origin) @ rotation
    if np.any(abs(floor_local[:2]) > [.7, .4]):
        return _invalid('bin_floor_not_over_registered_table')
    # One projected pixel plus the 6mm line membership band; these are explicit
    # engineering assumptions, not a statistical calibration certificate.
    pixel_pitch = float(np.max(d[edge]) / min(k[0, 0], k[1, 1]))
    translation_bound = max(.006, pixel_pitch)
    angle_bound = max(math.radians(.5), math.atan2(2*translation_bound, second['span']))
    if angle_bound > math.radians(5):
        return _invalid('table_edge_angle_uncertainty_too_large')
    radius = math.sqrt(.7**2+.4**2+.73**2)
    registration_margin = translation_bound + 2*radius*math.sin(angle_bound/2)
    return dict(valid=True, model='erc_table_two_edge_rgbd_v1', frame='base_footprint',
                origin=origin.tolist(), rotation=rotation.tolist(), solids=_solid_bounds(),
                modeled_margin_m=MODELED_MARGIN, registration_margin_m=registration_margin,
                mesh_sha256=TABLE_MESH_SHA256,
                quality=dict(plane_points=len(p), edge_points=len(ep), plane_rms_m=plane_rms,
                             plane_normal=normal.tolist(), plane_offset_m=h,
                             edge_spans_m=[first['span'], second['span']],
                             edge_rms_m=[first['rms'], second['rms']],
                             edge_absolute_dot=dot, pixel_pitch_m=pixel_pitch,
                             translation_bound_m=translation_bound, angle_bound_rad=angle_bound,
                             rectangle_outside_fraction=float(np.mean(outside)),
                             observed_floor_to_table_m=float(floor @ normal-h)),
                assumptions=['Aligned calibrated optical-Z RGBD and depth-epoch base transform.',
                             'Detected bin rests on this stock table; floor is 10 mm above bin bottom.',
                             'Stock 1.4 x 0.8 x 0.04 m top and four 0.07 m corner legs.',
                             'Hidden CAD edges/legs are inferred from the visible perpendicular edges.',
                             'Pixel/edge angular bounds are engineering allowances, not calibration proof.',
                             'Caller reprojects for base motion and invalidates registration after scene impact.'])


class TableSceneObstacle:
    """Five finite expanded solids; compatible with PlaceSceneChecker.intersects."""

    def __init__(self, scene):
        if not isinstance(scene, dict) or scene.get('valid') is not True:
            raise ValueError('table scene is not registered')
        if scene.get('model') != 'erc_table_two_edge_rgbd_v1' or scene.get('frame') != 'base_footprint':
            raise ValueError('unknown table scene/frame')
        self.origin = _finite(scene['origin'], (3,), 'table origin')
        transform = np.eye(4); transform[:3, :3] = _finite(scene['rotation'], (3, 3), 'table rotation')
        _rigid(transform)
        self.rotation = transform[:3, :3]
        modeled = float(scene['modeled_margin_m'])
        registration = float(scene['registration_margin_m'])
        margin = modeled + registration
        if (not math.isfinite(modeled) or not math.isfinite(registration)
                or modeled < MODELED_MARGIN or registration < 0 or not MODELED_MARGIN <= margin <= .12):
            raise ValueError('invalid table margin')
        solids = scene.get('solids', [])
        if solids != _solid_bounds() or scene.get('mesh_sha256') != TABLE_MESH_SHA256:
            raise ValueError('table CAD bounds/hash mismatch')
        signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
        self.solids = []
        self.solid_triangles = []
        faces = np.asarray([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        for solid in solids:
            bounds = np.asarray(solid['bounds']) + np.array([[-margin]*3, [margin]*3])
            centre, half = bounds.mean(axis=0), (bounds[1]-bounds[0])/2
            corners = (centre+signs*half) @ self.rotation.T + self.origin
            self.solids.append((solid['name'], bounds, corners))
            self.solid_triangles.append(corners[faces])
        self.margin = margin
        self.last_intersection = None
        self._local_bounds = ExactLocalBounds()
        # Only this exact built-in owns a bounded validation cache.
        from erc_phase1_solution.geometry_array_checks import ExactRigidValidation
        self._rigid_validation = (ExactRigidValidation(_rigid)
                                  if type(self) is TableSceneObstacle else None)

    def intersects(self, surface, transform, watertight):
        from erc_phase1_solution.kinematics import oriented_box_intersects_triangles
        from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
        from erc_phase1_solution.geometry_array_checks import any_boolean, ExactRigidValidation
        original_surface = surface
        immutable = surface.snapshot() if type(surface) is ModelLocalMesh else None
        if immutable is not None and surface.matches(immutable[0]):
            # Exact handle construction already checked all facets; the
            # whole bytes owner and fresh metadata identify that proof.
            surface = immutable[0]
        else:
            immutable = None
            surface = np.asarray(surface, dtype=float)
            if surface.ndim != 3 or surface.shape[1:] != (3, 3) or not len(surface) or not np.all(np.isfinite(surface)):
                raise ValueError('invalid table collision surface')
        rigid = getattr(self, '_rigid_validation', None)
        transform = (rigid.validate(transform, _rigid) if type(rigid) is ExactRigidValidation
                     else _rigid(transform))
        cache = getattr(self, '_local_bounds', None)
        cached = (immutable if immutable is not None else
                  cache.capture(original_surface) if cache is not None else None)
        if cached is None:
            lo, hi = surface.min(axis=(0, 1)), surface.max(axis=(0, 1))
        else:
            surface, lo, hi = cached
        rotation = self.rotation.T @ transform[:3, :3]
        translation = self.rotation.T @ (transform[:3, 3]-self.origin)
        centre = rotation @ ((lo+hi)/2) + translation
        half = np.abs(rotation) @ ((hi-lo)/2)
        world = None
        self.last_intersection = None
        for name, bounds, corners in self.solids:
            if any_boolean(centre+half < bounds[0]) or any_boolean(centre-half > bounds[1]):
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
