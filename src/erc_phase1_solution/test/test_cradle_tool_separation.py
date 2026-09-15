"""Actual mesh predicates: certain separation, contact and contained surfaces."""
import math
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from erc_phase1_solution import shelf_cradle_geometry as geometry
from erc_phase1_solution.kinematics import _box_triangles


PALM = geometry.PALM_COLLISION_LINK
ROBOT = 'arm_left_3_link'


class RotatedChain:
    active_names = ('translation',)
    lower = np.array([-1.])
    upper = np.array([1.])

    def __init__(self, angle=math.pi/4):
        c, s = math.cos(angle), math.sin(angle)
        self.rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    def forward(self, joints):
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = joints[0] * self.rotation[:, 0] + [0., 0., 1.]
        return transform


def make_node(gap=.03):
    chain = RotatedChain()
    palm = np.array([[[-.1, -.15, -.15], [-.1, .15, -.15], [-.1, 0., .15]]])
    world = palm @ chain.rotation.T + [0., 0., 1.]
    robot = world + gap * chain.rotation[:, 0]
    surfaces = {ROBOT: robot}
    def local(aperture, fingers=None):
        return {PALM: palm + [aperture - .02, 0., 0.]}
    node = SimpleNamespace(chain=chain, carried_book_dimensions=np.array([.16, .02, .25]),
        carried_transition_samples=3, carried_shelf_margin=.02,
        _shelf_cradle_geometry=SimpleNamespace(local_surfaces=local, watertight={PALM: False}),
        _robot_self_collision=lambda *args, **kwargs: None,
        _watertight_collision_links=lambda: frozenset(),
        _world_collision_surfaces=lambda *args, **kwargs: surfaces,
        _cancel=threading.Event())
    return node, surfaces, world


def check(node, start=0., end=None, aperture=.02):
    return geometry.check_cradle_tool_sweep(node, [5., 5., 1.], [0.],
        [start], [start if end is None else end], None, aperture=aperture)


class CradleToolSeparationTests(unittest.TestCase):
    def test_overlapping_aabbs_with_strict_directional_gap_skip_detailed_mesh(self):
        node, surfaces, world = make_node()
        other = surfaces[ROBOT]
        self.assertTrue(np.all(world.max(axis=(0, 1)) >= other.min(axis=(0, 1))))
        self.assertTrue(np.all(other.max(axis=(0, 1)) >= world.min(axis=(0, 1))))
        with patch.object(geometry, 'triangle_meshes_intersect', wraps=geometry.triangle_meshes_intersect) as detailed:
            self.assertIsNone(check(node))
        self.assertEqual(detailed.call_count, 0)

    def test_same_separated_pair_agrees_with_original_detailed_path(self):
        node, _, _ = make_node()
        with patch.object(geometry, 'separated_on_axes', return_value=False), patch.object(
                geometry, 'triangle_meshes_intersect', wraps=geometry.triangle_meshes_intersect) as detailed:
            self.assertIsNone(check(node))
        self.assertGreater(detailed.call_count, 0)

    def test_gap_inside_engineering_margin_retains_original_detailed_check(self):
        node, _, _ = make_node(50e-6)
        with patch.object(geometry, 'triangle_meshes_intersect', wraps=geometry.triangle_meshes_intersect) as detailed:
            self.assertIsNone(check(node))
        self.assertGreater(detailed.call_count, 0)

    def test_exact_surface_contact_is_rejected(self):
        node, _, _ = make_node(0.)
        self.assertTrue(check(node).startswith('cradle_tool_robot_intersection:'))

    def test_tool_inside_closed_robot_mesh_is_rejected(self):
        node, surfaces, world = make_node()
        surfaces[ROBOT] = _box_triangles([.6, .6, .6]) + world.mean(axis=(0, 1))
        node._watertight_collision_links = lambda: frozenset({ROBOT})
        self.assertTrue(check(node).startswith('cradle_tool_robot_intersection:'))

    def test_new_pose_rechecks_vertices_after_previous_clear_result(self):
        node, _, _ = make_node()
        self.assertIsNone(check(node))
        self.assertTrue(check(node, start=.03).startswith('cradle_tool_robot_intersection:'))

    def test_new_aperture_rechecks_vertices_after_previous_clear_result(self):
        node, _, _ = make_node()
        self.assertIsNone(check(node))
        self.assertTrue(check(node, aperture=.05).startswith('cradle_tool_robot_intersection:'))

    def test_collision_between_clear_endpoints_remains_rejected(self):
        node, _, _ = make_node()
        node.carried_transition_samples = 5  # Include the contact pose at .03.
        self.assertIsNone(check(node, start=0.))
        self.assertIsNone(check(node, start=.06))
        self.assertTrue(check(node, start=0., end=.06).startswith('cradle_tool_robot_intersection:'))

    def test_cancellation_is_still_checked_before_geometry(self):
        node, _, _ = make_node()
        node._cancel.set()
        with patch.object(geometry, 'separated_on_axes') as separated:
            self.assertEqual(check(node, end=.06), 'cradle_tool_cancelled')
        separated.assert_not_called()


if __name__ == '__main__':
    unittest.main()
