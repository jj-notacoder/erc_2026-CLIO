"""DRAFT registered local empty shelf bounds; no dispatch or source edits.

Shelf lip is anchored to the observed number-marker front surface, using only
unchanged official relative geometry. Vertical floor/roof retain the explicit
upright supported-book assumption because digit centroid Z is not plate center.
The 10 mm normal uncertainty is a disclosed engineering allowance, not measured
covariance. Existing 15 mm margin, 200 mm lateral uncertainty, 10 mm roof
uncertainty, and 5 mm tool allowance all remain. Unknown adjacent objects and
full shelf-mesh collision certification remain outside this local model.
"""
import math
import numpy as np
from erc_phase1_solution.lift_first_extraction import _bay_values
from erc_phase1_solution.rigid_palm_preflight import LEFT_GRIPPER_COLLISION_LINKS

def _cancel(node):
    if node._cancel.is_set():
        raise RuntimeError('empty_shelf_cancelled')

def _forward_vertices(triangles, plane_point, inward, margin):
    """Vertices of triangles clipped to the occupied side of a shelf plane.

    Include edge/plane intersections, since checking only original vertices
    would miss a side or roof violation at a triangle's crossing point.
    """
    facets = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    distances = (facets-plane_point) @ inward + margin
    active = np.any(distances >= 0, axis=1)
    facets, distances = facets[active], distances[active]
    if not len(facets):
        return np.empty((0, 3))
    result = [facets[distances >= 0]]
    for index in range(3):
        other = (index+1) % 3
        first, last = distances[:, index], distances[:, other]
        crosses = first*last < 0
        fractions = first[crosses]/(first[crosses]-last[crosses])
        result.append(facets[crosses, index] + fractions[:, None]*(
            facets[crosses, other]-facets[crosses, index]))
    return np.concatenate(result, axis=0)



class EmptyShelfBounds:
    """Supplement robot/book checks with the measured local shelf opening.

    Shelf floor follows the same explicit upright supported-book assumption
    as lift-first planning.  This is a bounded local model, not a certificate
    for all unobserved shelf geometry or neighboring books.
    """
    def __init__(self, node, front, grasp, guard, bay, *, normal_uncertainty_m=.010):
        self.node, self.guard = node, guard
        # grasp is accepted only for compatibility with the bottom draft;
        # None is valid because shelf geometry never depends on a grasp pose.
        dimensions = np.asarray(node.carried_book_dimensions, dtype=float)
        if dimensions.shape != (3,) or not np.allclose(dimensions, [.16,.02,.25], rtol=0., atol=1e-12):
            raise ValueError('empty_shelf_requires_official_book_dimensions')
        self.marker, self.left, self.side_min, self.side_max = _bay_values(bay)
        self.inward = np.asarray(bay.inward_axis_base, dtype=float)
        # Official shelf STL front is .244984 m outward from its origin;
        # marker origin is .245 m outward, with a .001 m visible face.
        # Register only their relative geometry, never the simulator world pose.
        self.plane_point = self.marker + .001016*self.inward
        self.normal_uncertainty_m = float(normal_uncertainty_m)
        if not np.isfinite(self.normal_uncertainty_m) or not .010 <= self.normal_uncertainty_m <= .050:
            raise ValueError('empty_shelf_normal_uncertainty_outside_bound')
        front = np.asarray(front, dtype=float)
        if front.shape != (3,) or not np.all(np.isfinite(front)):
            raise ValueError('empty_shelf_front_shape')
        # Marker glyph vertical position is not the card center. Preserve the
        # explicitly supported-book floor reference, not a fabricated row Z.
        if bay.assume_upright_supported is not True:
            raise ValueError('empty_shelf_floor_requires_supported_book')
        self.registered_lip = self.plane_point.copy()
        book_depth = float((front-self.registered_lip) @ self.inward)
        if not .015 <= book_depth <= .150:
            raise ValueError('empty_shelf_marker_book_depth_inconsistent')
        # Preserve the earlier nominal book-derived exclusion as an additional
        # conservative bound; it cannot authorize motion past the marker plane.
        old_plane = front - .065*self.inward
        earlier_shift = min(0., float((old_plane-self.plane_point) @ self.inward))
        self.plane_point += earlier_shift*self.inward
        self.book_depth_m = book_depth
        self.margin = float(bay.margin_m)
        self.floor = float(front[2])-.5*float(node.carried_book_dimensions[2])+self.margin
        self.roof = float(front[2])-.5*float(node.carried_book_dimensions[2])+.300-bay.roof_uncertainty_m-self.margin
        geometry = guard.geometry
        self.local = geometry.local_surfaces(guard.open_aperture)
        if set(self.local) != set(LEFT_GRIPPER_COLLISION_LINKS):
            raise ValueError('empty_tool_inventory')
        self.samples = 0
        self.minimum_floor = self.minimum_roof = self.minimum_side = self.minimum_back = math.inf

    def sample(self, q, allow_entry):
        _cancel(self.node)
        transform = self.node.chain.forward(q)
        surfaces = [(name, vertices, 0.) for name, vertices in
            self.node._world_collision_surfaces(q, **self.guard.context).items()]
        surfaces.extend((name, triangles @ transform[:3, :3].T+transform[:3, 3], .005)
                        for name, triangles in self.local.items())
        self.samples += 1
        for name, triangles, allowance in surfaces:
            triangles = np.asarray(triangles, dtype=float)
            if not np.all(np.isfinite(triangles)):
                return 'empty_shelf_nonfinite_surface:'+name
            occupied = _forward_vertices(triangles, self.plane_point,
                                          self.inward, self.margin+self.normal_uncertainty_m+allowance)
            if not len(occupied):
                continue
            if not allow_entry:
                return 'empty_setup_enters_shelf:'+name
            lateral = (occupied-self.marker) @ self.left
            floor = float(occupied[:, 2].min()-self.floor-allowance)
            roof = float(self.roof-occupied[:, 2].max()-allowance)
            side = float(min(lateral.min()-self.side_min-allowance,
                             self.side_max-lateral.max()-allowance))
            # Official clear depth: front STL z=.244984 to inner back z=-.055016.
            depth = (occupied-self.registered_lip) @ self.inward
            back = float(.300-self.margin-self.normal_uncertainty_m-allowance-depth.max())
            self.minimum_back = min(self.minimum_back, back)
            self.minimum_floor = min(self.minimum_floor, floor)
            self.minimum_roof = min(self.minimum_roof, roof)
            self.minimum_side = min(self.minimum_side, side)
            if min(floor, roof, side, back) < 0:
                return ('empty_bay_clearance:'+name
                        +f':floor={floor}:roof={roof}:side={side}:back={back}')
        return None

    def edge(self, start, end, *, allow_entry):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if start.shape != (8,) or end.shape != (8,) or not np.all(np.isfinite([start, end])):
            raise ValueError('empty_shelf_edge_shape')
        count = max(61, int(self.node.carried_transition_samples),
            int(math.ceil(float(np.max(np.abs(end-start)))/.02))+1)
        for fraction in np.linspace(0., 1., count):
            reason = self.sample(start+(end-start)*fraction, allow_entry)
            if reason:
                return reason
        return None


_ShelfEntryBounds = EmptyShelfBounds
