"""Complete-tool guard rejects hazards omitted by the arm collision model."""
from types import SimpleNamespace

import numpy as np

from erc_phase1_solution import shelf_cradle_geometry as geometry


PALM = geometry.PALM_COLLISION_LINK


class TranslationChain:
    active_names = ('translation',)
    lower = np.asarray([-1.])
    upper = np.asarray([1.])

    def forward(self, joints):
        transform = np.eye(4)
        transform[0, 3] = joints[0]
        transform[2, 3] = 1.
        return transform


def node_with_surface(palm_x=-.1, robot=None):
    palm = np.asarray([[[palm_x, -.1, -.1], [palm_x, .1, -.1], [palm_x, 0., .1]]])
    model = SimpleNamespace(
        local_surfaces=lambda *_: {PALM: palm}, watertight={PALM: False},
    )
    return SimpleNamespace(
        chain=TranslationChain(), carried_book_dimensions=np.asarray([.16, .03, .25]),
        carried_transition_samples=3, carried_shelf_margin=.02,
        gripper_transport_lock=.03, _shelf_cradle_geometry=model,
        _robot_self_collision=lambda _: None,
        _watertight_collision_links=lambda: frozenset(),
        _world_collision_surfaces=lambda _: robot or {},
    )


def check(node, start=(0.,), end=(0.,), plane=1.):
    return geometry.check_cradle_tool_sweep(node, [0., 0., 1.], [0.], start, end, plane)


def test_unpadded_book_palm_penetration_is_rejected():
    assert check(node_with_surface(palm_x=.05)) == 'cradle_book_palm_intersection'
    assert check(node_with_surface()) is None


def test_tool_robot_intersection_between_endpoints_is_rejected():
    obstacle = np.asarray([[[.1, -.1, .9], [.1, .1, .9], [.1, 0., 1.1]]])
    node = node_with_surface(robot={'arm_left_3_link': obstacle})
    assert check(node, end=(.4,)).startswith('cradle_tool_robot_intersection:')


def test_tool_shelf_margin_is_preserved():
    assert 'cradle_tool_shelf_clearance' in check(node_with_surface(), plane=-.09)


def test_geometry_error_fails_closed():
    assert check(node_with_surface(), end=(float('nan'),)) == 'cradle_tool_joint_nonfinite'


def test_body_self_collision_also_blocks_tool_route():
    node = node_with_surface()
    node._robot_self_collision = lambda _: ('arm_left_3_link', 'torso_lift_link')
    assert check(node).startswith('cradle_robot_self_intersection:')


def test_all_body_meshes_must_clear_the_measured_shelf_plane():
    obstacle = np.asarray([[[.99, -.1, .9], [.99, .1, .9], [.99, 0., 1.1]]])
    node = node_with_surface(robot={'arm_left_3_link': obstacle})
    assert check(node).startswith('cradle_robot_shelf_clearance:')


def test_arm_cannot_pass_through_floor_even_without_shelf_plane():
    low_arm = np.asarray([[[.5, -.1, .01], [.5, .1, .01], [.5, 0., .03]]])
    node = node_with_surface(robot={'arm_left_3_link': low_arm})
    assert check(node, plane=None).startswith('cradle_robot_floor_clearance:')


def test_hard_joint_limits_are_preserved():
    assert check(node_with_surface(), end=(1.1,)) == 'cradle_tool_joint_limit'


def test_route_merges_straight_sections_but_retains_reversal(monkeypatch):
    seen = []
    monkeypatch.setattr(geometry, 'check_cradle_tool_sweep', lambda *args, **kwargs: seen.append((args[3].tolist(), args[4].tolist())))
    assert geometry.check_cradle_tool_route(
        object(), [0, 0, 0], [0], [0], [[.1], [.2], [.3], [.1]], 1.,
    ) is None
    assert seen == [([0.], [.3]), ([.3], [.1])]


def test_opening_checks_intermediate_finger_collisions(monkeypatch):
    def collide_in_middle(*args, **kwargs):
        if .03 <= kwargs['aperture'] <= .04:
            return 'cradle_tool_robot_intersection'
        return None

    monkeypatch.setattr(geometry, 'check_cradle_tool_sweep', collide_in_middle)
    assert 'cradle_tool_robot_intersection' in geometry.check_gripper_opening(
        object(), [0, 0, 0], [0], [0], 1.,
    )
