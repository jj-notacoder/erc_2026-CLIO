"""Portable synthetic geometry/coordinator cases; no ROS, assets or world poses."""
from types import SimpleNamespace
import itertools
import math
import threading

import numpy as np
import pytest

from erc_phase1_solution.kinematics import _box_triangles
from erc_phase1_solution.scene_checked_place import (
    NominalBinObstacle, PlaceSceneChecker, coordinated_supported_goals, certified_bin_cavity,
)
from erc_phase1_solution.rigid_palm_preflight import PALM_COLLISION_LINK


BOUNDS = np.array([[-.155, -.105, -.305], [.155, .105, .255]])
CAVITY = np.array([[-.145, -.095, -.245], [.145, .105, .245]])


def transform(position=(0., 0., 0.), rotation=None):
    result = np.eye(4)
    result[:3, 3] = position
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def corners(size=(.05, .02, .06)):
    return np.array(list(itertools.product((-1., 1.), repeat=3))) * np.asarray(size)/2


@pytest.mark.parametrize('point', ([1., 0., .75], [.8, -.6, .9], [-.3, .8, .4]))
def test_bin_ray_floor_registration_and_full_asymmetric_bounds(point):
    raw = BOUNDS.copy()
    obstacle = NominalBinObstacle(point, raw)
    ray = np.r_[point[:2], 0.]
    ray /= np.linalg.norm(ray)
    assert np.allclose(obstacle.rotation[:, 2], ray)
    assert np.array_equal(obstacle.rotation[:, 1], [0., 0., 1.])
    assert np.allclose(obstacle.rotation.T @ obstacle.rotation, np.eye(3))
    assert np.linalg.det(obstacle.rotation) == pytest.approx(1.)
    assert np.allclose(obstacle.origin + obstacle.rotation @ [0., -.095, 0.], point)
    local = (obstacle.corners-obstacle.origin) @ obstacle.rotation
    assert np.allclose(local.min(axis=0), raw[0]-.005)
    assert np.allclose(local.max(axis=0), raw[1]+.005)
    assert np.array_equal(raw, BOUNDS)
    raw[:] = 99.
    assert np.allclose(obstacle.bounds, BOUNDS + [[-.005]*3, [.005]*3])


@pytest.mark.parametrize('point,bounds,margin', [
    ([0., 0., .75], BOUNDS, .005),
    ([float('nan'), 0., .75], BOUNDS, .005),
    ([1., .75], BOUNDS, .005),
    ([1., 0., .75], [[0, 0, 0], [0, 1, 1]], .005),
    ([1., 0., .75], BOUNDS, -.001),
    ([1., 0., .75], BOUNDS, float('inf')),
])
def test_invalid_nominal_obstacle_rejected(point, bounds, margin):
    with pytest.raises(ValueError):
        NominalBinObstacle(point, bounds, margin=margin)


def test_obstacle_detects_crossing_triangle_without_inside_vertices():
    obstacle = NominalBinObstacle([.8, .6, .75], BOUNDS)
    triangle = np.array([[[-1., 0., -1.], [1., 0., -1.], [0., 0., 1.]]])
    assert obstacle.intersects(triangle, transform(obstacle.origin, obstacle.rotation), False)


def test_obstacle_detects_mesh_inside_box_and_box_inside_closed_mesh():
    obstacle = NominalBinObstacle([1., 0., .75], BOUNDS)
    pose = transform(obstacle.origin, obstacle.rotation)
    assert obstacle.intersects(_box_triangles([.04, .04, .04]), pose, True)
    enclosing = _box_triangles([2., 2., 2.])
    assert obstacle.intersects(enclosing, pose, True)
    assert not obstacle.intersects(enclosing, pose, False)


def test_disjoint_and_margin_only_collision():
    plain = NominalBinObstacle([1., 0., .75], BOUNDS, margin=0.)
    padded = NominalBinObstacle([1., 0., .75], BOUNDS, margin=.005)
    # The tiny box has a 2 mm nominal gap to the +X bin face.
    position = plain.origin + plain.rotation @ [.158, 0., 0.]
    mesh = _box_triangles([.002, .002, .002])
    pose = transform(position, plain.rotation)
    assert not plain.intersects(mesh, pose, True)
    assert padded.intersects(mesh, pose, True)
    assert not padded.intersects(mesh, transform(position+[0., 2., 0.]), True)


def test_hollow_bin_accepts_cavity_but_rejects_each_wall_floor_and_margin():
    obstacle = NominalBinObstacle([1., 0., .75], BOUNDS, cavity_bounds=CAVITY)
    assert len(obstacle.material_bounds) == 5
    mesh = _box_triangles([.002]*3)
    for local, expected in [([0., 0., 0.], False), ([0., .2, 0.], False),
            ([-.15, 0., 0.], True), ([.15, 0., 0.], True),
            ([0., 0., -.28], True), ([0., 0., .25], True),
            ([0., -.10, 0.], True), ([.141, 0., 0.], True)]:
        position = obstacle.origin + obstacle.rotation @ local
        assert obstacle.intersects(mesh, transform(position, obstacle.rotation), True) is expected
        book = corners([.002]*3) @ obstacle.rotation.T + position
        assert obstacle.book_intersects(book) is expected


def test_hollow_slabs_detect_mesh_enclosing_wall_and_book_crossing_cavity_boundary():
    obstacle = NominalBinObstacle([1., 0., .75], BOUNDS, cavity_bounds=CAVITY)
    assert obstacle.intersects(_box_triangles([2., 2., 2.]),
                               transform(obstacle.origin, obstacle.rotation), True)
    book = corners([.31, .05, .10]) @ obstacle.rotation.T + obstacle.origin
    assert obstacle.book_intersects(book)


def test_cavity_certificate_refuses_face_crossing_void_with_all_vertices_outside():
    obstacle = NominalBinObstacle([1., 0., .75], BOUNDS, cavity_bounds=CAVITY, margin=0.)
    # Local closed slabs form a conservative surrogate for the known CAD.
    material = np.concatenate([(surface-obstacle.origin) @ obstacle.rotation
                               for surface in obstacle.material_triangles])
    assert np.array_equal(certified_bin_cavity(material), CAVITY)
    intruder = np.array([[[-1., 0., -1.], [1., 0., -1.], [0., 0., 1.]]])
    assert certified_bin_cavity(np.concatenate([material, intruder])) is None


class SignedChain:
    """Proper rotations with R[2,1]=cos(q7-wanted(q1))."""
    def __init__(self, wanted=lambda q: -2.2+.5*q[1]):
        self.wanted = wanted
        self.lower = np.array([0.] + [-3.]*7)
        self.upper = np.array([.35] + [3.]*7)

    def forward(self, q):
        angle = float(q[-1])-self.wanted(q)+np.pi/2
        c, s = np.cos(angle), np.sin(angle)
        result = np.eye(4)
        result[:3, :3] = [[1., 0., 0.], [0., c, -s], [0., s, c]]
        return result


def endpoints():
    start, goal = np.zeros(8), np.zeros(8)
    start[0] = goal[0] = .30
    start[-1] = -2.2
    goal[1], goal[-1] = 1., -1.75
    return start, goal


@pytest.mark.parametrize('power', (1., .5))
def test_coordinator_signed_continuity_exact_endpoint_and_nonmutation(power):
    chain = SignedChain()
    start, goal = endpoints()
    original_start, original_goal = start.copy(), goal.copy()
    result = coordinated_supported_goals(chain, start, goal, margin=.10,
                                         minimum_support=.75, shoulder_progress_power=power)
    assert result is not None and len(result) == 21
    assert np.array_equal(result[-1], goal)  # The IK endpoint is not replaced by the optimum.
    assert np.array_equal(start, original_start) and np.array_equal(goal, original_goal)
    values = np.array(result)
    assert np.all(values[:, 1:] >= chain.lower[1:]+.10)
    assert np.all(values[:, 1:] <= chain.upper[1:]-.10)
    assert np.all(values[:, -1] < 0.)
    assert max(abs(np.diff(np.r_[start[-1], values[:, -1]]))) < .2
    assert all(chain.forward(q)[2, 1] > .99 for q in result)
    assert values[0, 1] == pytest.approx(.05**power)
    result[-1][1] = 9.
    assert np.array_equal(goal, original_goal)


def test_coordinator_can_raise_second_and_fifth_joints_before_other_joints():
    start, goal = endpoints()
    goal[2:7] = [.8, -.6, -1.5, .5, 1.2]
    powers = np.asarray([1., .5, 1., 1., .5, 1.])
    result = coordinated_supported_goals(SignedChain(), start, goal, margin=.10,
        minimum_support=.75, joint_progress_powers=powers)
    assert result is not None and np.array_equal(result[-1], goal)
    for fraction, q in zip(np.linspace(.05, 1., 20), result):
        np.testing.assert_allclose(q[1:7], start[1:7]+(goal-start)[1:7]*fraction**powers)
        assert SignedChain().forward(q)[2, 1] >= .75
    np.testing.assert_array_equal(powers, [1., .5, 1., 1., .5, 1.])


@pytest.mark.parametrize('powers', [[], [1.]*5, [1.]*7, [1., 1., float('nan'), 1., 1., 1.], [.25]*6])
def test_invalid_per_joint_progress_policy_rejected(powers):
    with pytest.raises(ValueError):
        coordinated_supported_goals(SignedChain(), *endpoints(), margin=.10,
            minimum_support=.75, joint_progress_powers=powers)


def test_coordinator_does_not_accept_upside_down_absolute_support():
    chain = SignedChain(lambda q: np.pi)
    chain.lower[-1], chain.upper[-1] = -.30, .30
    start = np.zeros(8); start[0] = .30
    goal = start.copy(); goal[1] = .2
    assert abs(chain.forward(start)[2, 1]) > .99
    assert chain.forward(start)[2, 1] < -.99
    assert coordinated_supported_goals(chain, start, goal, margin=.10, minimum_support=.75) is None


def test_coordinator_refuses_disconnected_signed_roll_branch():
    chain = SignedChain(lambda q: 2.8 if q[1] < .5 else -2.8)
    start, goal = endpoints()
    start[-1], goal[-1] = 2.8, -2.8
    assert coordinated_supported_goals(chain, start, goal, margin=.10, minimum_support=.75) is None


@pytest.mark.parametrize('which,index,value', [('start', 1, -2.95), ('goal', 7, 2.95), ('start', 0, -.001)])
def test_coordinator_preserves_arm_margin_and_separate_torso_limits(which, index, value):
    start, goal = endpoints()
    (start if which == 'start' else goal)[index] = value
    assert coordinated_supported_goals(SignedChain(), start, goal, margin=.10, minimum_support=.75) is None


@pytest.mark.parametrize('kwargs', [dict(margin=-1., minimum_support=.75),
    dict(margin=float('nan'), minimum_support=.75), dict(margin=.1, minimum_support=0.),
    dict(margin=.1, minimum_support=1.01), dict(margin=.1, minimum_support=.75, shoulder_progress_power=.75)])
def test_invalid_coordinator_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        coordinated_supported_goals(SignedChain(), *endpoints(), **kwargs)


def scene_fixture(*, robot_link='arm_left_6_link', tool_link='gripper_left_outer_finger_left_link'):
    node = SimpleNamespace(_cancel=threading.Event(), _lock=threading.RLock(),
                           carried_transition_samples=3)
    node.right, node.head = np.zeros(7), np.zeros(2)
    node._resolved_right_positions = lambda _: node.right.copy()
    node._resolved_head_positions = lambda _: node.head.copy()
    node.robot_pose = transform([0., .5, 1.])
    node.hand_pose = transform([0., 0., 1.2])
    node.contexts = []
    def robot_transforms(q, **context):
        node.contexts.append({k: v.copy() for k, v in context.items()})
        return {robot_link: node.robot_pose.copy()}
    node._collision_link_transforms = robot_transforms
    node.chain = SimpleNamespace(forward=lambda q: node.hand_pose.copy())
    node.carried_collision_meshes = [SimpleNamespace(link=robot_link,
        triangles=_box_triangles([.04, .04, .04]), watertight=True)]
    node._robot_self_collision = lambda q, **context: None
    node._watertight_collision_links = lambda: {robot_link}
    tool = SimpleNamespace(local_surfaces=lambda aperture: {tool_link: _box_triangles([.02, .02, .02])},
                           watertight={tool_link: True})
    obstacle = NominalBinObstacle([1., 0., .75], BOUNDS)
    checker = PlaceSceneChecker(node, obstacle, tool, corners())
    q = np.zeros(8); q[0] = .30
    return checker, node, q


def test_scene_cache_keys_all_context_and_aperture_and_loaded_exactly():
    scene, node, q = scene_fixture()
    assert scene.sample(q, .017, False)
    assert scene.sample(q.copy(), .017, False) and scene.cache_hits == 1
    for action in ('right', 'head', 'q', 'aperture', 'loaded'):
        if action == 'right': node.right[0] += 1e-12
        if action == 'head': node.head[0] += 1e-12
        if action == 'q': q[1] += 1e-12
        assert scene.sample(q, .069 if action == 'aperture' else .017, action == 'loaded')
    assert len(node.contexts) == 6 and scene.cache_hits == 1
    assert node.contexts[-1]['right_positions'][0] == 1e-12
    assert node.contexts[-1]['head_positions'][0] == 1e-12


def test_cancel_checked_before_a_cached_scene_pass():
    scene, node, q = scene_fixture()
    assert scene.sample(q, .017, False)
    node._cancel.set()
    assert not scene.sample(q, .017, False)
    assert scene.last_rejection['reason'] == 'cancelled'


@pytest.mark.parametrize('target,reason', [('robot', 'robot_bin'), ('tool', 'tool_bin'), ('book', 'book_bin')])
def test_scene_checks_distinct_robot_tool_and_loaded_book(target, reason):
    scene, node, q = scene_fixture()
    if target == 'robot': node.robot_pose = transform(scene.obstacle.origin)
    if target == 'tool': node.hand_pose = transform(scene.obstacle.origin)
    if target == 'book':
        node.carried_collision_meshes = []
        scene.tool.local_surfaces = lambda aperture: {}
        node.hand_pose = transform(scene.obstacle.origin)
        assert scene.sample(q, .017, False)  # Empty return has no fictional attached book.
    assert not scene.sample(q, .017, True)
    assert scene.last_rejection['reason'] == reason


def test_open_tool_self_check_uses_measured_robot_context():
    scene, node, q = scene_fixture()
    node._robot_self_collision = lambda q, **context: ('arm_left_4_link', 'torso_lift_link')
    assert not scene.sample(q, .069, False)
    assert scene.last_rejection == {'reason': 'robot_self', 'pair': ['arm_left_4_link', 'torso_lift_link']}


@pytest.mark.parametrize('target', ['robot', 'tool', 'book'])
def test_screen_visual_box_rejects_arm_tool_and_held_book(target):
    scene, node, q = scene_fixture()
    if target == 'robot':
        centre = node.robot_pose[:3, 3]
    else:
        centre = node.hand_pose[:3, 3]
    if target == 'book':
        scene.tool.local_surfaces = lambda aperture: {}
    scene.screen = SimpleNamespace(world=lambda torso, head: corners([.01, .01, .01])+centre)
    assert not scene.sample(q, .017, True)
    assert scene.last_rejection['reason'] == ('book_head_screen' if target == 'book' else 'head_screen')


@pytest.mark.parametrize('robot_link,tool_link,expected', [
    ('arm_left_7_link', PALM_COLLISION_LINK, True),
    ('arm_left_7_link', 'gripper_left_outer_finger_left_link', False),
    ('arm_left_6_link', PALM_COLLISION_LINK, False),
])
def test_tool_robot_exclusion_is_only_palm_wrist_adjacency(robot_link, tool_link, expected):
    scene, node, q = scene_fixture(robot_link=robot_link, tool_link=tool_link)
    node.robot_pose = node.hand_pose.copy()
    assert scene.sample(q, .069, False) is expected
    if not expected:
        assert scene.last_rejection['reason'] == 'tool_robot'


@pytest.mark.parametrize('which,reason', [('robot', 'left_arm_ground'), ('tool', 'tool_ground')])
def test_moving_left_and_open_tool_ground_guard(which, reason):
    scene, node, q = scene_fixture()
    if which == 'robot': node.robot_pose[2, 3] = .025
    else: node.hand_pose[2, 3] = .015
    assert not scene.sample(q, .069, False)
    assert scene.last_rejection['reason'] == reason


def test_scene_leg_checks_interior_and_both_endpoints_with_joint_sampling_bound():
    scene, _, first = scene_fixture()
    last = first.copy(); last[1] = .08
    seen = []
    def sample(q, aperture, loaded):
        seen.append(q.copy())
        assert aperture == .069 and loaded is False
        return True
    scene.sample = sample
    assert scene.leg(first, last, .069, False)
    assert np.array_equal(seen[0], first) and np.array_equal(seen[-1], last)
    assert len(seen) >= 5 and max(np.diff(np.array(seen)[:, 1])) <= .020000000001
    seen.clear()
    assert scene.leg(first, first.copy(), .069, False) and len(seen) == 1


def test_scene_leg_does_not_accept_safe_endpoints_with_blocked_interior():
    scene, _, first = scene_fixture()
    last = first.copy(); last[1] = .08
    def sample(q, aperture, loaded):
        if .019 < q[1] < .061:
            return scene._reject('synthetic_interior_collision')
        return True
    scene.sample = sample
    assert not scene.leg(first, last, .017, True)
    assert scene.last_rejection['reason'] == 'synthetic_interior_collision'
    assert 0 < scene.last_rejection['fraction'] < 1


@pytest.mark.parametrize('field', ('q', 'aperture', 'attached'))
def test_scene_rejects_nonfinite_inputs(field):
    scene, node, q = scene_fixture()
    with pytest.raises(ValueError):
        if field == 'attached':
            invalid = corners(); invalid[0, 0] = float('nan')
            PlaceSceneChecker(node, scene.obstacle, scene.tool, invalid)
        else:
            if field == 'q': q[2] = float('nan')
            scene.sample(q, float('nan') if field == 'aperture' else .017, False)
