"""ROS-aware tests for graceful shutdown after the Humble context is invalid."""

from __future__ import annotations

from importlib import import_module
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


rclpy = pytest.importorskip('rclpy')
ExternalShutdownException = import_module(
    'rclpy.executors'
).ExternalShutdownException
manipulation_node = import_module('erc_phase1_solution.manipulation_node')
mission_manager = import_module('erc_phase1_solution.mission_manager')
navigation_node = import_module('erc_phase1_solution.navigation_node')
perception_node = import_module('erc_phase1_solution.perception_node')
empty_pickup = import_module('erc_phase1_solution.empty_pickup_collision')


class _WorkflowEmptyPickupGuard:
    """Protocol recorder for fixtures whose joint vectors are command sentinels.

    These values are deliberately not physical poses or sensor measurements.
    Real collision/freshness policies remain exercised in test_empty_pickup_collision;
    this fake checks the production PICK call order and rejection propagation.
    """
    def __init__(self, node, failure=None):
        self.node, self.failure = node, failure
        self.start, self.right, self.head = np.zeros(8), np.zeros(7), np.zeros(2)
        self.initial_aperture, self.open_aperture = .018, node.gripper_open
        self.checked_samples = self.cache_hits = 0
        self.last_rejection = None
        self.calls = []
        self.fresh_count = 0

    def _accept(self, name, *values):
        self.calls.append((name, *values))
        if self.failure == name:
            self.last_rejection = 'fixture_' + name
            return False
        return True

    def opening(self):
        return self._accept('opening')

    def edge(self, first, last):
        return self._accept('edge', np.asarray(first).copy(), np.asarray(last).copy())

    def retracted_edge(self, first, last):
        return self._accept('retracted_edge', np.asarray(first).copy(), np.asarray(last).copy())

    def plan_transition(self, first, last):
        return [] if self._accept('plan_transition', np.asarray(first).copy(), np.asarray(last).copy()) else None

    def candidate(self, solutions, transition):
        return self._accept('candidate', [np.asarray(q).copy() for q in solutions],
                            [np.asarray(q).copy() for q in transition])

    def require_fresh(self, expected, aperture):
        self.fresh_count += 1
        name = 'fresh_' + str(self.fresh_count)
        if not self._accept(name, np.asarray(expected).copy(), float(aperture)):
            raise RuntimeError('fixture_' + name)

    def wait_for_endpoint(self, previous, expected, *, aperture, phase):
        name = 'wait_' + phase
        if not self._accept(name, np.asarray(previous).copy(),
                            np.asarray(expected).copy(), float(aperture)):
            raise RuntimeError('fixture_' + name)
        return np.asarray(expected).copy()


def _invalidate_context(monkeypatch, module):
    shutdown_calls = []
    monkeypatch.setattr(module.rclpy, 'init', lambda args=None: None)
    monkeypatch.setattr(module.rclpy, 'ok', lambda: False)
    monkeypatch.setattr(module.rclpy, 'shutdown', lambda: shutdown_calls.append(True))
    return shutdown_calls


@pytest.mark.parametrize(
    ('module', 'node_class_name', 'has_zero_command'),
    (
        (perception_node, 'PerceptionNode', False),
        (navigation_node, 'NavigationNode', True),
    ),
)
@pytest.mark.parametrize('shutdown_error', (ExternalShutdownException, RuntimeError))
def test_single_threaded_main_avoids_ros_calls_after_external_shutdown(
    monkeypatch, module, node_class_name, has_zero_command, shutdown_error
):
    node = SimpleNamespace(destroyed=False, zero_commands=0)
    node.destroy_node = lambda: setattr(node, 'destroyed', True)
    node._publish_zero = lambda: setattr(
        node, 'zero_commands', node.zero_commands + 1
    )
    shutdown_calls = _invalidate_context(monkeypatch, module)
    if module is perception_node:
        monkeypatch.setattr(module.cv2, 'setNumThreads', lambda count: None)
    monkeypatch.setattr(module, node_class_name, lambda: node)
    monkeypatch.setattr(
        module.rclpy,
        'spin',
        lambda unused: (_ for _ in ()).throw(shutdown_error()),
    )

    module.main()

    assert node.destroyed
    assert shutdown_calls == []
    if has_zero_command:
        assert node.zero_commands == 0


@pytest.mark.parametrize('shutdown_error', (ExternalShutdownException, RuntimeError))
def test_manipulation_main_avoids_double_shutdown(monkeypatch, shutdown_error):
    node = SimpleNamespace(
        _cancel=SimpleNamespace(was_set=False),
        destroyed=False,
    )
    node._cancel.set = lambda: setattr(node._cancel, 'was_set', True)
    node.destroy_node = lambda: setattr(node, 'destroyed', True)

    executor = SimpleNamespace(node=None, stopped=False)
    executor.add_node = lambda added: setattr(executor, 'node', added)
    executor.spin = lambda: (_ for _ in ()).throw(shutdown_error())
    executor.shutdown = lambda: setattr(executor, 'stopped', True)

    shutdown_calls = _invalidate_context(monkeypatch, manipulation_node)
    monkeypatch.setattr(manipulation_node, 'ManipulationNode', lambda: node)
    monkeypatch.setattr(
        manipulation_node,
        'SingleThreadedExecutor',
        lambda: executor,
    )

    manipulation_node.main()

    assert node._cancel.was_set
    assert executor.node is node
    assert executor.stopped
    assert node.destroyed
    assert shutdown_calls == []


def _official_manipulation_planner():
    source_root = Path(__file__).resolve().parents[2]
    urdf = (
        source_root
        / 'erc_description'
        / 'urdf'
        / 'tiago_pro.urdf'
    )
    if not urdf.exists():
        pytest.skip('official erc_description URDF is not in this workspace')
    node = object.__new__(manipulation_node.ManipulationNode)
    node.chain = manipulation_node.URDFChain.from_urdf(
        urdf,
        'base_footprint',
        'gripper_left_grasping_link',
        manipulation_node.IK_JOINTS,
    )
    node.right_chain = manipulation_node.URDFChain.from_urdf(
        urdf,
        'base_footprint',
        'gripper_right_grasping_link',
        ('torso_lift_joint', *manipulation_node.RIGHT_ARM_JOINTS),
    )
    node.head_chain = manipulation_node.URDFChain.from_urdf(
        urdf,
        'base_footprint',
        'head_front_camera_link',
        ('torso_lift_joint', 'head_1_joint', 'head_2_joint'),
    )
    package_paths = {
        'omni_base_description': (
            source_root / 'omni_base_robot' / 'omni_base_description'
        ),
        'tiago_pro_description': (
            source_root / 'tiago_pro_robot' / 'tiago_pro_description'
        ),
        'tiago_pro_head_description': (
            source_root / 'tiago_pro_head_robot' / 'tiago_pro_head_description'
        ),
        'pal_sea_arm_description': (
            source_root / 'pal_sea_arm' / 'pal_sea_arm_description'
        ),
    }
    node.carried_collision_meshes = manipulation_node.load_urdf_collision_meshes(
        urdf,
        manipulation_node.CARRIED_COLLISION_LINKS,
        package_paths.__getitem__,
    )
    node.position_tolerance = 0.012
    node.orientation_tolerance = 0.10
    node.place_joint_limit_margin = 0.03
    node.cartesian_clearance = 0.45
    node.top_row_cartesian_clearance = 0.46
    node.top_row_grasp_vertical_offset = -0.015
    node.top_row_grasp_lateral_offset = -0.001
    node.top_row_loaded_clearance_lift = 0.018
    node.cartesian_step = 0.06
    node.cartesian_joint_step = 0.40
    node.retreat_distance = 0.18
    node.carried_book_dimensions = np.asarray([0.16, 0.03, 0.25])
    node.carried_book_padding = 0.015
    node.carried_shelf_margin = 0.02
    node.carried_shelf_retreat_clearance = 0.25
    node.carried_cradle_retraction = 0.05
    node.carried_cradle_extension = 0.12
    node.carried_cradle_roll = -1.10
    node.carried_cradle_transfer = 0.75
    node.carried_supported_jaw_vertical_component = 0.75
    node.carried_transition_samples = 61
    node.carried_orientation_step_limit = 0.45
    node.carried_navigation_radius_limit = 0.45
    node.carried_maximum_tilt = math.pi / 3.0
    node.joints = {
        'head_1_joint': 0.0,
        'head_2_joint': 0.0,
        **dict(zip(manipulation_node.RIGHT_ARM_JOINTS, manipulation_node.RIGHT_HOME)),
    }
    node._self_collision_cache = {}
    node._static_self_collision_cache = {}
    node._current_seed = lambda: manipulation_node.HOME.copy()
    return node


def _head_follow_test_node(preflight_safe):
    """Use actual shared admission; only controller futures and geometry are fake."""
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._cancel = threading.Event()
    node._lock = threading.Lock()
    node._goal_handles = []
    node.timeout = 1.0
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000))
    node._publish_status = lambda *args, **kwargs: None
    calls = []
    node._carried_head_transition_is_safe = (
        lambda pan, tilt: calls.append(('preflight', pan, tilt)) or preflight_safe
    )
    result = SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(
        status=manipulation_node.GoalStatus.STATUS_SUCCEEDED))
    handle = SimpleNamespace(accepted=True, get_result_async=lambda: result)
    def send(goal):
        calls.append(('send', goal.trajectory.joint_names,
                      list(goal.trajectory.points[0].positions)))
        return SimpleNamespace(done=lambda: True, result=lambda: handle)
    node.head_client = SimpleNamespace(wait_for_server=lambda **kwargs: True, send_goal_async=send)
    return node, calls


def test_held_head_motion_uses_collision_preflight():
    node, calls = _head_follow_test_node(True)
    assert node._move_head(0.1, -0.28)
    assert calls == [('preflight', 0.1, -0.28),
                     ('send', list(manipulation_node.HEAD_JOINTS), [0.1, -0.28])]
    assert node._goal_handles == []


def test_held_head_motion_rejects_unsafe_sweep_before_controller_call():
    node, calls = _head_follow_test_node(False)
    with pytest.raises(RuntimeError, match='blocks requested head motion'):
        node._move_head(0.0, -0.28)
    assert calls == [('preflight', 0.0, -0.28)]
    assert node._goal_handles == []


def test_collision_transforms_use_measured_right_arm_and_fail_closed():
    node = object.__new__(manipulation_node.ManipulationNode)
    right_positions = manipulation_node.RIGHT_HOME + np.asarray(
        [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01]
    )
    node.joints = {
        'head_1_joint': 0.0,
        'head_2_joint': -0.28,
        **dict(zip(manipulation_node.RIGHT_ARM_JOINTS, right_positions)),
    }
    node.chain = SimpleNamespace(link_transforms=lambda unused: {})
    recorded_right = []
    node.right_chain = SimpleNamespace(
        link_transforms=lambda positions: recorded_right.append(
            np.asarray(positions, dtype=float)
        )
        or {
            link: np.eye(4)
            for link in manipulation_node.RIGHT_COLLISION_LINKS
        }
    )
    node.head_chain = SimpleNamespace(
        link_transforms=lambda unused: {
            link: np.eye(4)
            for link in manipulation_node.HEAD_COLLISION_LINKS
        }
    )

    node._collision_link_transforms(np.zeros(8))

    assert len(recorded_right) == 1
    assert recorded_right[0] == pytest.approx(
        np.concatenate(([0.0], right_positions))
    )
    del node.joints[manipulation_node.RIGHT_ARM_JOINTS[-1]]
    with pytest.raises(RuntimeError, match='without right arm joints'):
        node._collision_link_transforms(np.zeros(8))


def test_self_collision_cache_key_tracks_measured_right_arm():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.joints = {
        'head_1_joint': 0.0,
        'head_2_joint': -0.28,
        **dict(
            zip(
                manipulation_node.RIGHT_ARM_JOINTS,
                manipulation_node.RIGHT_HOME,
            )
        ),
    }
    node._self_collision_cache = {}
    node._static_self_collision_cache = {}
    surface_calls = []
    node._world_collision_surfaces = lambda solution, **kwargs: (
        surface_calls.append(kwargs) or {}
    )
    solution = manipulation_node.CARRY.copy()

    assert node._robot_self_collision(solution) is None
    assert node._robot_self_collision(solution) is None
    node.joints[manipulation_node.RIGHT_ARM_JOINTS[0]] += 0.01
    assert node._robot_self_collision(solution) is None

    assert len(surface_calls) == 2
    assert surface_calls[0]['right_positions'][0] == pytest.approx(
        manipulation_node.RIGHT_HOME[0]
    )
    assert surface_calls[1]['right_positions'][0] == pytest.approx(
        manipulation_node.RIGHT_HOME[0] + 0.01
    )


def test_carried_transition_rejects_an_interior_self_collision():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.cartesian_joint_step = 0.40
    node.carried_orientation_step_limit = 0.45
    node.carried_transition_samples = 3
    node.carried_maximum_tilt = math.pi / 3.0
    node.carried_supported_jaw_vertical_component = 0.80
    node.chain = SimpleNamespace(
        forward=lambda unused: np.eye(4),
        pose_error=lambda first, last: np.zeros(6),
    )
    node._carried_robot_collision = lambda *args, **kwargs: None
    checked = []

    def self_collision(solution):
        checked.append(np.asarray(solution, dtype=float).copy())
        if np.isclose(solution[1], 0.20, atol=1e-12):
            return 'arm_left_2_link', 'torso_lift_link'
        return None

    node._robot_self_collision = self_collision
    start = np.zeros(8)
    end = start.copy()
    end[1] = 0.40
    corners = np.asarray(
        [
            (x, y, z)
            for x in (-0.10, 0.10)
            for y in (-0.02, 0.02)
            for z in (-0.15, 0.15)
        ]
    )

    assert not node._carried_robot_transition_is_safe(start, end, corners)
    assert any(np.isclose(solution[1], 0.20) for solution in checked)


def test_carried_route_can_require_gravity_support_independently_of_attitude():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.cartesian_joint_step = 0.40
    node.carried_orientation_step_limit = 0.45
    node.carried_transition_samples = 5
    node.carried_supported_jaw_vertical_component = 0.82
    node.chain = SimpleNamespace(
        forward=lambda unused: np.eye(4),
        pose_error=lambda first, last: np.zeros(6),
    )
    node._carried_robot_transition_is_safe = lambda *args: True
    start = np.zeros(8)
    goal = start.copy()
    goal[1] = 0.10
    corners = np.zeros((8, 3))

    assert node._plan_carried_joint_route(start, [goal], corners) is not None
    assert node._plan_carried_joint_route(
        start,
        [goal],
        corners,
        require_gravity_support=True,
    ) is None


def _top_row_pick_plan(node, front):
    front = np.asarray(front, dtype=float)
    grasp = front + np.asarray([0.060, 0.0, 0.0])
    grasp[1] += node.top_row_grasp_lateral_offset
    grasp[2] += node.top_row_grasp_vertical_offset
    pregrasp = grasp - np.asarray([0.14, 0.0, 0.0])
    clearance = pregrasp.copy()
    clearance[0] = node.top_row_cartesian_clearance
    clearance[2] += node.top_row_loaded_clearance_lift
    positions = [clearance]
    positions.extend(node._interpolate_positions(clearance, pregrasp, 0.06))
    positions.extend(node._interpolate_positions(pregrasp, grasp, 0.06))
    rotations = manipulation_node.shelf_pinch_orientations(grasp[2])
    solutions, orientation_index, score, transition = (
        node._solve_cartesian_path(
            positions,
            rotations,
            0.35,
            endpoint_first=True,
        )
    )
    return grasp, positions, rotations, solutions, orientation_index, score, transition


@pytest.mark.parametrize(
    'front',
    (
        np.asarray([0.717, -0.066, 1.583]),
        np.asarray([0.692739, -0.062894, 1.583684]),
        np.asarray([0.692936, -0.056937, 1.583779]),
    ),
)
def test_top_row_endpoint_first_pick_uses_audited_yaw_free_branch(front):
    node = _official_manipulation_planner()
    grasp, positions, _, solutions, orientation_index, score, transition = (
        _top_row_pick_plan(node, front)
    )

    assert len(solutions) == len(positions)
    assert orientation_index == 0
    assert np.isfinite(score)
    assert all(solution[0] == pytest.approx(0.35) for solution in solutions)
    assert np.linalg.norm(node.chain.forward(solutions[-1])[:3, 3] - grasp) <= 0.012
    # OFFER-seeded endpoint solving must retain the audited elbow-up family,
    # not the wrist-below-shelf branch that caused the live contacts.  Check
    # branch invariants rather than a brittle distance from one target's IK
    # snapshot: the competing branch has arm1 > pi/2 and arm2/arm6 < 0.
    endpoint = solutions[-1]
    assert endpoint[1] < math.pi / 2.0
    assert endpoint[2] > 0.0
    assert endpoint[6] > 0.0

    current = manipulation_node.HOME.copy()
    current[0] = 0.35
    route = [current, *transition, solutions[0]]
    changed_joints = []
    for first, second in zip(route, route[1:]):
        assert node._retracted_transition_is_safe(first, second)
        changed = np.flatnonzero(
            ~np.isclose(first[1:], second[1:], atol=1e-10)
        )
        assert len(changed) == 1
        changed_joints.append(int(changed[0]))
    assert changed_joints == list(range(7))


def test_top_row_depth_is_shallower_without_changing_lower_rows():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.grasp_depth_offset = 0.08
    node.top_row_grasp_depth_offset = 0.060
    node.top_row_grasp_vertical_offset = -0.015
    node.top_row_grasp_lateral_offset = -0.001
    node.top_row_loaded_clearance_lift = 0.018

    assert node._grasp_depth_for_height(1.583) == pytest.approx(0.060)
    assert node._grasp_depth_for_height(1.419) == pytest.approx(0.08)


def test_zero_transport_lock_accepts_bounded_servo_undershoot_with_contact():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._lock = None
    node.joints = {'gripper_left_finger_joint': -1.5e-7}
    node.gripper_closed = 0.0
    node.gripper_transport_lock = 0.0
    node.grasp_min_position = 0.014
    node.grasp_min_margin = 0.001
    node.grasp_max_position = 0.05
    node._transport_lock_engaged = True
    node._gravity_supported_payload = False
    node._target_contact_sides = lambda **kwargs: (True, True)

    verified, width, left, right, plausible = node._pinch_sample()

    assert verified
    assert width == pytest.approx(-1.5e-7)
    assert left and right
    assert plausible


def test_cartesian_place_regression_for_aligned_bin_pose():
    node = _official_manipulation_planner()
    bin_surface = np.asarray([0.72, -0.068, 0.752])
    release = bin_surface + np.asarray([0.17, 0.0, 0.22])
    above = release + np.asarray([0.0, 0.0, 0.13])
    clearance = above.copy()
    clearance[0] = node.cartesian_clearance
    positions = [clearance]
    positions.extend(node._interpolate_positions(clearance, above, 0.06))
    positions.extend(node._interpolate_positions(above, release, 0.06))

    solutions, _, _, transition = node._solve_cartesian_path(
        positions,
        manipulation_node.bin_place_orientations(),
        0.30,
    )

    assert len(solutions) == len(positions)
    assert all(solution[0] == pytest.approx(0.30) for solution in solutions)
    assert np.linalg.norm(node.chain.forward(solutions[-1])[:3, 3] - release) <= 0.012
    current = manipulation_node.HOME.copy()
    current[0] = 0.30
    route = [current, *transition, solutions[0]]
    assert all(
        node._retracted_transition_is_safe(first, second)
        for first, second in zip(route, route[1:])
    )


@pytest.mark.parametrize(
    'front',
    (
        np.asarray([0.717, -0.066, 1.583]),
        np.asarray([0.692739, -0.062894, 1.583684]),
        # Trial d3b9c4dae6a3 live top-row perception target.
        np.asarray([0.6937779125, -0.05493238067, 1.58418506676]),
        # Trial 1ebe29e3a1c7 live top-row perception target.
        np.asarray([0.69273995043, -0.05796243404, 1.58368404285]),
    ),
)
def test_top_row_retreat_then_cradle_and_supported_return_is_safe(front):
    node = _official_manipulation_planner()
    _, positions, rotations, solutions, orientation_index, _, _ = _top_row_pick_plan(
        node,
        front,
    )

    (
        lowering,
        cradle_route,
        cradle_roll_index,
        transport_route,
        planned_corners,
    ) = node._plan_carried_return(
        front,
        solutions[-1],
        solutions[1],
        rotations[orientation_index],
        0.35,
    )

    # The shelf-side pick ends in an elevated vertical pinch.  The last two
    # extraction legs raise the target by 12 mm, unloading the top-row shelf
    # before the fixed-heading base retreat.  Every attitude-changing motion
    # remains deferred, so no q7 sweep can place the book on arm_left_5 beside
    # the shelf.
    assert lowering == []
    assert cradle_route == []
    assert cradle_roll_index is None
    assert transport_route == []
    shelf_start = np.asarray(solutions[1], dtype=float)
    assert positions[1][2] - positions[-1][2] == pytest.approx(0.012)
    assert (
        node.chain.forward(shelf_start)[2, 3]
        - node.chain.forward(solutions[-1])[2, 3]
    ) >= 0.009

    cached_plan = node._cached_post_retreat_plan
    assert cached_plan is not None
    np.testing.assert_allclose(cached_plan['start'], shelf_start)
    assert cached_plan['requires_gravity_support'] is False
    assert cached_plan['shelf_front_x'] == pytest.approx(
        front[0] + node.carried_shelf_retreat_clearance
    )
    cached_phases = [phase for _, phase in cached_plan['legs']]
    assert cached_phases == (
        ['post_retreat_clearance_extension'] * 2
        + ['post_retreat_cradle_roll'] * 3
        + ['supported_cradle_lowering'] * 3
        + ['supported_cradle_retraction']
        + ['compact_transport'] * 20
    )

    # First move the still-vertical pinch 0.12 m away from the arm, then roll
    # only q7.  The local roll keeps the grasp origin and links 1--6 fixed.
    expected_extension = node._solve_post_retreat_clearance_extension(
        shelf_start
    )
    assert len(expected_extension) == 2
    for (actual, _), expected in zip(
        cached_plan['legs'][:2], expected_extension
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    extended = np.asarray(expected_extension[-1], dtype=float)
    cradle = node._solve_carried_cradle(extended)
    validated_cradle = node._plan_carried_joint_route(
        extended,
        [cradle],
        planned_corners,
        post_retreat_shelf_front_x=cached_plan['shelf_front_x'],
    )
    assert validated_cradle is not None
    assert len(validated_cradle) == 3
    for (actual, _), expected in zip(
        cached_plan['legs'][2:5], validated_cradle
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(cradle[:-1], extended[:-1], atol=1e-12)
    assert cradle[-1] - extended[-1] == pytest.approx(
        node.carried_cradle_roll
    )
    assert node.chain.forward(cradle)[2, 1] >= (
        node.carried_supported_jaw_vertical_component
    )
    shelf_pose = node.chain.forward(shelf_start)
    extended_pose = node.chain.forward(extended)
    np.testing.assert_allclose(
        extended_pose[:3, 3],
        shelf_pose[:3, 3] + np.asarray([0.12, 0.0, 0.0]),
        atol=node.position_tolerance,
    )
    np.testing.assert_allclose(
        extended_pose[:3, :3],
        shelf_pose[:3, :3],
        atol=node.orientation_tolerance,
    )

    # Once q7 has loaded the lower jaw, lower 0.18 m and retract only 0.05 m.
    # Most of the forward clearance remains during staging, keeping the book
    # away from the link-5 sweep that caused the live shelf-side failure.
    extension, expected_lowering, expected_retraction = (
        node._solve_supported_post_retreat_staging(cradle)
    )
    assert extension == []
    assert len(expected_lowering) == 3
    assert len(expected_retraction) == 1
    for (actual, _), expected in zip(
        cached_plan['legs'][5:8],
        expected_lowering,
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(
        cached_plan['legs'][8][0],
        expected_retraction[-1],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        cached_plan['staging_terminal'],
        expected_retraction[-1],
        atol=1e-10,
    )
    start_pose = node.chain.forward(cradle)
    lowered_pose = node.chain.forward(expected_lowering[-1])
    np.testing.assert_allclose(
        lowered_pose[:3, 3],
        start_pose[:3, 3] + np.asarray([0.0, 0.0, -0.18]),
        atol=node.position_tolerance,
    )
    assert all(
        solution[0] == pytest.approx(0.35)
        for solution in (*expected_lowering, *expected_retraction)
    )
    retracted_pose = node.chain.forward(expected_retraction[-1])
    assert retracted_pose[0, 3] == pytest.approx(
        start_pose[0, 3] - node.carried_cradle_retraction,
        abs=node.position_tolerance,
    )

    # Every cached leg respects both the payload shelf plane and the complete
    # modeled-robot shelf plane after retreat and keeps the lower jaw upward.
    # Only the final SUPPORTED_CARRY posture must fit the navigation footprint;
    # the base remains stationary throughout the preceding interpolation.
    post_retreat_front_x = front[0] + node.carried_shelf_retreat_clearance
    previous = shelf_start
    for waypoint, phase in cached_plan['legs']:
        assert node._carried_post_retreat_transition_is_safe(
            previous,
            waypoint,
            planned_corners,
            post_retreat_front_x,
        )
        if phase in (
            'supported_cradle_lowering',
            'supported_cradle_retraction',
            'compact_transport',
        ):
            assert node._gravity_supported_transition_is_safe(previous, waypoint)
        previous = waypoint
    expected_carry = manipulation_node.SUPPORTED_CARRY.copy()
    expected_carry[0] = 0.35
    np.testing.assert_allclose(cached_plan['terminal'], expected_carry)
    np.testing.assert_allclose(previous, expected_carry)
    assert node._carried_navigation_radius(
        expected_carry,
        planned_corners,
    ) <= node.carried_navigation_radius_limit


def test_supported_return_has_bounded_robot_safe_reverse_bin_route():
    node = _official_manipulation_planner()
    front = np.asarray([0.69273995043, -0.05796243404, 1.58368404285])
    _, _, rotations, pick_solutions, orientation_index, _, _ = (
        _top_row_pick_plan(node, front)
    )
    (
        lowering,
        cradle_route,
        cradle_roll_index,
        transport_route,
        attached_corners,
    ) = node._plan_carried_return(
        front,
        pick_solutions[-1],
        pick_solutions[1],
        rotations[orientation_index],
        0.35,
    )
    assert lowering == []
    assert cradle_route == []
    assert cradle_roll_index is None
    assert transport_route == []
    cached_plan = node._cached_post_retreat_plan
    assert cached_plan is not None
    post_retreat_front_x = front[0] + node.carried_shelf_retreat_clearance
    previous = np.asarray(cached_plan['start'], dtype=float)
    for waypoint, phase in cached_plan['legs']:
        assert node._carried_post_retreat_transition_is_safe(
            previous,
            waypoint,
            attached_corners,
            post_retreat_front_x,
        )
        if phase in (
            'supported_cradle_lowering',
            'supported_cradle_retraction',
            'compact_transport',
        ):
            assert node._gravity_supported_transition_is_safe(previous, waypoint)
        previous = np.asarray(waypoint, dtype=float)
    staging = np.asarray(cached_plan['staging_terminal'], dtype=float).copy()
    transport = np.asarray(cached_plan['terminal'], dtype=float).copy()
    np.testing.assert_allclose(previous, transport, atol=1e-12)
    node._current_seed = lambda: transport.copy()

    bin_surface = np.asarray([0.72, -0.068, 0.752])
    release = bin_surface + np.asarray([0.17, 0.0, 0.22])
    above = release + np.asarray([0.0, 0.0, 0.13])
    clearance = above.copy()
    clearance[0] = node.cartesian_clearance
    positions = [clearance]
    positions.extend(node._interpolate_positions(clearance, above, 0.06))
    positions.extend(node._interpolate_positions(above, release, 0.06))
    torso_ready = transport.copy()
    torso_ready[0] = 0.30
    place_staging = staging.copy()
    place_staging[0] = 0.30
    assert node._gravity_supported_transition_is_safe(transport, torso_ready)
    forward_supported = node._supported_compact_goals(place_staging)
    carried_staging_waypoints = [
        *(waypoint.copy() for waypoint in reversed(forward_supported[:-1])),
        place_staging,
    ]
    carried_staging_route = node._plan_carried_joint_route(
        torso_ready,
        carried_staging_waypoints,
        attached_corners,
        require_gravity_support=True,
    )
    assert carried_staging_route is not None
    candidate_start = carried_staging_route[-1]

    def carried_candidate_is_safe(candidate_solutions, candidate_transition):
        candidate_route = node._plan_carried_joint_route(
            candidate_start,
            (*candidate_transition, candidate_solutions[0]),
            attached_corners,
            require_gravity_support=True,
        )
        if candidate_route is None:
            return False
        previous = candidate_route[-1]
        for candidate_solution in candidate_solutions[1:]:
            if not node._carried_robot_transition_is_safe(
                previous,
                candidate_solution,
                attached_corners,
            ):
                return False
            if not node._gravity_supported_transition_is_safe(
                previous,
                candidate_solution,
            ):
                return False
            previous = candidate_solution
        return True

    solutions, place_orientation_index, _, transition = node._solve_cartesian_path(
        positions,
        manipulation_node.supported_bin_place_orientations(),
        0.30,
        transition_start=place_staging,
        candidate_validator=carried_candidate_is_safe,
        first_valid=True,
        joint_limit_margin=max(node.place_joint_limit_margin, 0.10),
    )
    assert place_orientation_index == 0

    candidate_route = node._plan_carried_joint_route(
        candidate_start,
        (*transition, solutions[0]),
        attached_corners,
        require_gravity_support=True,
    )
    assert candidate_route is not None
    route = [*carried_staging_route, *candidate_route]
    assert np.allclose(route[-1], solutions[0])
    previous = torso_ready
    for waypoint in route:
        assert node._carried_robot_transition_is_safe(
            previous,
            waypoint,
            attached_corners,
        )
        assert node._gravity_supported_transition_is_safe(previous, waypoint)
        previous = waypoint
    for solution in solutions[1:]:
        assert node._carried_robot_transition_is_safe(
            previous,
            solution,
            attached_corners,
        )
        assert node._gravity_supported_transition_is_safe(previous, solution)
        previous = solution
    home_at_place_height = manipulation_node.HOME.copy()
    home_at_place_height[0] = 0.30
    unloaded_home_waypoints = node._plan_retracted_transition(
        torso_ready,
        home_at_place_height,
    )
    assert unloaded_home_waypoints is not None
    unloaded_route = [
        torso_ready,
        *unloaded_home_waypoints,
        home_at_place_height,
    ]
    assert all(
        node._retracted_transition_is_safe(first, last)
        for first, last in zip(unloaded_route, unloaded_route[1:])
    )


def test_empty_close_recovery_reverses_cartesian_then_transition_route():
    node = object.__new__(manipulation_node.ManipulationNode)
    moved = []
    node.arm_client = object()
    node._move_arm_solution = lambda solution, duration: moved.append(
        ('arm', int(np.asarray(solution)[1]), duration)
    ) or True
    node._follow = lambda client, names, positions, duration: moved.append(
        ('home', tuple(positions), duration)
    ) or True
    node._move_torso = lambda height, duration, **kwargs: moved.append(
        ('torso', height, duration)
    ) or True
    cartesian = [np.full(8, value, dtype=float) for value in (10, 11, 12)]
    transition = [np.full(8, value, dtype=float) for value in (20, 21)]

    assert node._recover_unloaded_pick(cartesian, transition)
    assert [entry[:2] for entry in moved] == [
        ('arm', 11),
        ('arm', 10),
        ('arm', 21),
        ('arm', 20),
        ('home', tuple(manipulation_node.HOME[1:])),
        ('torso', float(manipulation_node.HOME[0])),
    ]


def test_bin_return_retraces_carry_then_uses_forward_unloaded_home_route():
    node = object.__new__(manipulation_node.ManipulationNode)
    moved = []
    node.arm_client = object()
    node._move_arm_solution = lambda solution, duration: moved.append(
        ('arm', int(np.asarray(solution)[1]), duration)
    ) or True
    node._follow = lambda client, names, positions, duration: moved.append(
        ('home', tuple(positions), duration)
    ) or True
    node._move_torso = lambda height, duration, **kwargs: moved.append(
        ('torso', height, duration)
    ) or True
    cartesian = [np.full(8, value, dtype=float) for value in (10, 11, 12)]
    carried_transition = [
        np.full(8, value, dtype=float) for value in (20, 21)
    ]
    carried_start = np.full(8, 5, dtype=float)
    unloaded_home = [np.full(8, value, dtype=float) for value in (30, 31)]

    assert node._return_from_bin(
        cartesian,
        carried_transition,
        carried_start,
        unloaded_home,
    )
    assert [entry[:2] for entry in moved] == [
        ('arm', 11),
        ('arm', 10),
        ('arm', 21),
        ('arm', 20),
        ('arm', 5),
        ('arm', 30),
        ('arm', 31),
        ('home', tuple(manipulation_node.HOME[1:])),
        ('torso', float(manipulation_node.HOME[0])),
    ]


def test_carried_shelf_retreat_preserves_heading_and_moves_along_normal():
    node = object.__new__(mission_manager.MissionManager)
    node.robot_pose = (1.0, 2.0, 0.0)
    node.shelf_normal = np.asarray([0.6, 0.8])
    node.carried_shelf_retreat = 0.35

    x, y, yaw = node._carried_shelf_retreat_goal()

    assert (x, y) == pytest.approx((1.21, 2.28))
    assert yaw == pytest.approx(0.0)


def test_profiled_navigation_is_one_atomic_command():
    node = object.__new__(mission_manager.MissionManager)
    node.nav_event = {'event': 'stale'}
    node.nav_goals = 0
    node.latest_planned_path = object()
    node.latest_executed_path = object()
    node.active_nav_purpose = ''
    node.active_nav_goal_number = 0
    node.active_nav_dispatched_ns = 0
    node.active_nav_request = {}
    node.active_nav_terminal = object()
    node.active_nav_pending = False
    node.trial_id = 'test'
    clock_time = import_module('rclpy.time').Time(seconds=1.0)
    node.get_clock = lambda: SimpleNamespace(now=lambda: clock_time)
    normal_goals = []
    commands = []
    node.nav_goal_pub = SimpleNamespace(
        topic_name='/erc/navigation/goal',
        publish=lambda message: normal_goals.append(message),
    )
    node.nav_command_pub = SimpleNamespace(
        topic_name='/erc/navigation/command',
        publish=lambda message: commands.append(message),
    )
    node._log = lambda *args, **kwargs: None

    node._navigate(
        1.25,
        -0.40,
        0.15,
        'carried_shelf_retreat',
        profile='carried_retreat',
    )

    assert normal_goals == []
    assert len(commands) == 1
    payload = mission_manager.decode_event(commands[0].data)
    assert payload['event'] == 'navigate'
    assert payload['profile'] == 'carried_retreat'
    assert (payload['x'], payload['y'], payload['yaw']) == pytest.approx(
        (1.25, -0.40, 0.15)
    )
    assert node.active_nav_pending
    assert node.active_nav_request['profile'] == 'carried_retreat'


@pytest.mark.parametrize(
    ('robot_pose', 'shelf_normal', 'reason'),
    (
        ((float('nan'), 2.0, 0.0), np.asarray([0.6, 0.8]), 'robot pose'),
        ((1.0, 2.0, float('nan')), np.asarray([0.6, 0.8]), 'robot heading'),
        ((1.0, 2.0, 0.0), np.asarray([float('nan'), 0.8]), 'shelf normal'),
    ),
)
def test_carried_shelf_retreat_rejects_nonfinite_geometry(
    robot_pose,
    shelf_normal,
    reason,
):
    node = object.__new__(mission_manager.MissionManager)
    node.robot_pose = robot_pose
    node.shelf_normal = shelf_normal
    node.carried_shelf_retreat = 0.70

    with pytest.raises(ValueError, match=reason):
        node._carried_shelf_retreat_goal()


def test_mission_compacts_only_after_straight_carried_retreat():
    node = object.__new__(mission_manager.MissionManager)
    node.finished = False
    node.wall_started = time.monotonic()
    node.trial_timeout = 1200.0
    node.state = 'CLEAR_SHELF_WITH_BOOK'
    node.state_started = None
    node.navigation_timeout = 50.0
    node.manipulation_timeout = 75.0
    node.manip_event = None
    node.start_pose = (2.0, 3.0, 0.2)
    node.shelf_normal = np.asarray([0.6, 0.8])
    node._elapsed_state = lambda: 0.0
    node._nav_reached = lambda: True
    node._manip_failed = lambda: False
    node._nav_failed = lambda: False
    calls = []
    node._manipulate = lambda command: calls.append(('manipulate', command))
    node._navigate = lambda x, y, yaw, purpose: calls.append(
        ('navigate', x, y, yaw, purpose)
    )
    node._set_state = lambda state, **fields: calls.append(
        ('state', state, fields)
    )
    node._abort = lambda reason: (_ for _ in ()).throw(
        AssertionError(f'unexpected abort: {reason}')
    )

    node._tick()

    assert calls == [
        ('manipulate', 'compact_transport'),
        ('state', 'COMPACT_TRANSPORT', {}),
    ]

    node.state = 'COMPACT_TRANSPORT'
    node.manip_event = {'event': 'succeeded', 'command': 'compact_transport'}
    calls.clear()
    node._tick()

    assert calls[0][0] == 'navigate'
    assert calls[0][1:3] == pytest.approx((2.0, 3.0))
    assert calls[0][3] == pytest.approx(math.atan2(0.8, 0.6))
    assert calls[0][4] == 'return_start_with_book'
    assert calls[1] == ('state', 'RETURN_START', {})


def test_mission_aborts_confirmed_payload_loss_without_retrying_pick():
    node = object.__new__(mission_manager.MissionManager)
    node.finished = False
    node.wall_started = time.monotonic()
    node.trial_timeout = 1200.0
    node.state = 'PICK'
    node.state_started = None
    node.manipulation_timeout = 75.0
    node.pick_attempts = 1
    node.max_pick_attempts = 3
    node.manip_event = {
        'event': 'failed',
        'command': 'pick',
        'reason': 'pick_payload_lost',
    }
    node._elapsed_state = lambda: 0.0
    node._manip_succeeded = lambda command: False
    node._manip_failed = lambda: True
    calls = []
    node._abort = lambda reason: calls.append(('abort', reason))
    node._manipulate = lambda command: calls.append(('manipulate', command))
    node._perception_mode = lambda mode: calls.append(('perception', mode))
    node._set_state = lambda state, **fields: calls.append(('state', state))

    node._tick()

    assert calls == [('abort', 'pick_payload_lost')]


@pytest.mark.parametrize('state', ('CLEAR_SHELF_WITH_BOOK', 'PLACE'))
def test_mission_aborts_immediately_on_asynchronous_payload_hazard(state):
    node = object.__new__(mission_manager.MissionManager)
    node.state = state
    node.finished = False
    node.manip_event = {'event': 'started', 'command': 'place'}
    calls = []
    node._log = lambda event, **fields: calls.append(('log', event, fields))
    node._abort = lambda reason: calls.append(('abort', reason))
    message = manipulation_node.String()
    message.data = mission_manager.encode_event(
        'payload_hazard',
        reason='contact_lost',
    )

    node._on_manipulation_status(message)

    assert calls[-1] == ('abort', 'payload_hazard:contact_lost')
    assert node.manip_event == {'event': 'started', 'command': 'place'}


def test_mission_records_the_concrete_book_identity_latched_by_the_gripper():
    node = object.__new__(mission_manager.MissionManager)
    node.target_colour = 'red'
    node.target_physical_column = 4
    node.detected_row = 1
    node.target_book_model = None
    node.manip_event = None
    node._log = lambda *args, **kwargs: None
    message = manipulation_node.String()
    message.data = mission_manager.encode_event(
        'target_book_latched',
        model='book_col_4_row_2_red',
    )

    node._on_manipulation_status(message)

    assert node.target_book_model == 'book_col_4_row_2_red'
    assert node.manip_event is None


def test_mission_rejects_identity_changes_outside_the_active_pick():
    node = object.__new__(mission_manager.MissionManager)
    node.target_colour = 'red'
    node.target_physical_column = 2
    node.detected_row = 1
    node.target_book_model = 'book_col_2_row_1_red'
    node.state = 'PLACE'
    node.manip_event = None
    node._log = lambda *args, **kwargs: None
    message = manipulation_node.String()
    message.data = mission_manager.encode_event(
        'target_book_latched',
        model='book_col_4_row_1_red',
    )

    node._on_manipulation_status(message)

    assert node.target_book_model == 'book_col_2_row_1_red'
    assert node.manip_event is None


def test_mission_clears_previous_identity_before_dispatching_a_retry_pick():
    node = object.__new__(mission_manager.MissionManager)
    node.finished = False
    node.wall_started = time.monotonic()
    node.trial_timeout = 1200.0
    node.state = 'REACQUIRE_BOOK'
    node.book_point = SimpleNamespace(header=SimpleNamespace(
        stamp=SimpleNamespace(sec=1, nanosec=0)))
    node.pick_attempts = 1
    node.target_book_model = 'book_col_2_row_1_red'
    calls = []
    node._manipulate = lambda command: calls.append(('manipulate', command))
    node._set_state = lambda state, **fields: calls.append(
        ('state', state, fields)
    )

    node._tick()

    assert node.pick_attempts == 2
    assert node.target_book_model is None
    assert calls == [
        ('manipulate', 'pick'),
        ('state', 'PICK', {}),
    ]


def test_compact_transport_checks_radius_and_retention_before_success():
    node = object.__new__(manipulation_node.ManipulationNode)
    _stub_retained_leg_dispatch(node)
    node._held_book_corners = np.zeros((8, 3))
    node._post_retreat_shelf_front_x = 0.95
    node._gravity_supported_payload = False
    node.carried_navigation_radius_limit = 0.45
    current = np.zeros(8)
    compact_goal = manipulation_node.CARRY.copy()
    compact_goal[0] = current[0]
    measured_terminal = compact_goal.copy()
    measured_terminal[1] += 0.001
    measured = iter((current.copy(), measured_terminal.copy()))
    node._measured_left_solution = lambda: next(measured)
    node._fresh_retention_probe = lambda *args, **kwargs: True
    node._target_contact_recent = lambda **kwargs: True
    node._supported_compact_goals = lambda *args: (_ for _ in ()).throw(
        AssertionError('lower-row fallback must retain ordinary CARRY')
    )
    safe_poses = []
    node._carried_robot_transition_is_safe = lambda *args: (_ for _ in ()).throw(
        AssertionError('post-retreat compaction must retain the shelf constraint')
    )
    node._carried_post_retreat_transition_is_safe = (
        lambda start, end, corners, shelf_front_x: safe_poses.append(
            (
                np.asarray(start).copy(),
                np.asarray(end).copy(),
                shelf_front_x,
            )
        ) or True
    )
    planned_goals = []
    planned_shelf_fronts = []

    def plan_carried_route(
        start,
        goals,
        corners,
        *,
        post_retreat_shelf_front_x=None,
        require_gravity_support=False,
    ):
        planned_goals.append(np.asarray(goals[0]).copy())
        planned_shelf_fronts.append(post_retreat_shelf_front_x)
        return [np.asarray(goals[0]).copy()]

    node._plan_carried_joint_route = plan_carried_route

    def navigation_radius(solution, corners):
        if np.allclose(solution, current):
            return 0.70
        if np.allclose(solution, measured_terminal):
            return 0.439
        return 0.4363

    node._carried_navigation_radius = navigation_radius
    moved = []
    node._move_arm_solution = lambda solution, duration: moved.append(
        (np.asarray(solution), duration)
    ) or True
    status = []
    node._publish_status = lambda event, **fields: status.append((event, fields))

    assert node._compact_transport()

    assert len(moved) == 1
    assert len(planned_goals) == 1
    np.testing.assert_allclose(planned_goals[0], compact_goal)
    assert moved[0][0] == pytest.approx(compact_goal)
    assert len(safe_poses) == 2
    assert safe_poses[0][0] == pytest.approx(current)
    assert safe_poses[0][1] == pytest.approx(current)
    assert safe_poses[0][2] == pytest.approx(0.95)
    assert safe_poses[1][0] == pytest.approx(measured_terminal)
    assert safe_poses[1][1] == pytest.approx(measured_terminal)
    assert safe_poses[1][2] == pytest.approx(0.95)
    assert planned_shelf_fronts == pytest.approx([0.95])
    assert node._post_retreat_shelf_front_x is None
    assert status[-1] == (
        'transport_compact',
        {
            'command': 'compact_transport',
            'waypoints': 1,
            'planar_radius': pytest.approx(0.439),
        },
    )


def test_compact_transport_selects_supported_goals_for_cradled_payload():
    node = object.__new__(manipulation_node.ManipulationNode)
    _stub_retained_leg_dispatch(node)
    node._held_book_corners = np.zeros((8, 3))
    node._post_retreat_shelf_front_x = 0.95
    node._gravity_supported_payload = True
    node.carried_navigation_radius_limit = 0.45
    current = np.zeros(8)
    supported_goals = [np.ones(8), np.full(8, 2.0)]
    measured = iter((current.copy(), supported_goals[-1].copy()))
    node._measured_left_solution = lambda: next(measured)
    requested_starts = []

    def supported(start):
        requested_starts.append(np.asarray(start).copy())
        return [goal.copy() for goal in supported_goals]

    node._supported_compact_goals = supported
    node._fresh_retention_probe = lambda *args, **kwargs: True
    node._target_contact_recent = lambda **kwargs: True
    node._carried_post_retreat_transition_is_safe = lambda *args: True
    node._gravity_supported_transition_is_safe = lambda *args: True
    node._carried_navigation_radius = lambda solution, corners: (
        0.70 if np.allclose(solution, current) else 0.436
    )
    planned = []

    def plan(start, goals, corners, **kwargs):
        planned.extend(np.asarray(goal).copy() for goal in goals)
        return [np.asarray(goal).copy() for goal in goals]

    node._plan_carried_joint_route = plan
    moved = []
    node._move_arm_solution = lambda solution, duration: moved.append(
        np.asarray(solution).copy()
    ) or True
    node._publish_status = lambda *args, **kwargs: None

    assert node._compact_transport()

    assert len(requested_starts) == 1
    np.testing.assert_allclose(requested_starts[0], current)
    assert len(planned) == len(supported_goals)
    assert len(moved) == len(supported_goals)
    for actual, expected in zip(planned, supported_goals):
        np.testing.assert_allclose(actual, expected)


def _minimal_post_retreat_cache(node, current):
    phases = [
        'post_retreat_clearance_extension',
        'post_retreat_cradle_roll',
        'supported_cradle_lowering',
        'supported_cradle_retraction',
        'compact_transport',
    ]
    goals = []
    for index in range(1, len(phases) + 1):
        goal = np.asarray(current, dtype=float).copy()
        goal[1:] = 0.1 * index
        goals.append(goal)
    return {
        'requires_gravity_support': False,
        'start': np.asarray(current, dtype=float).copy(),
        'shelf_front_x': 0.95,
        'attached_corners': node._held_book_corners.copy(),
        'legs': list(zip(goals, phases)),
        'staging_terminal': goals[3].copy(),
        'terminal': goals[-1].copy(),
        'compact_radius': 0.436,
    }


def test_compact_transport_executes_vertical_roll_supported_groups_continuously():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._post_retreat_shelf_front_x = 0.95
    node._gravity_supported_payload = False
    node._supported_post_retreat_staging_required = True
    node.carried_navigation_radius_limit = 0.45
    node.carried_cradle_transfer = 0.75
    current = np.zeros(8)
    phases = [
        'post_retreat_clearance_extension',
        'post_retreat_clearance_extension',
        'post_retreat_cradle_roll',
        'post_retreat_cradle_roll',
        'supported_cradle_lowering',
        'supported_cradle_retraction',
        'compact_transport',
        'compact_transport',
    ]
    goals = []
    for index in range(1, len(phases) + 1):
        goal = np.full(8, float(index), dtype=float)
        # The continuous goal commands only q[1:].  Every preflighted cached
        # point must therefore keep the measured torso coordinate unchanged.
        goal[0] = current[0]
        goals.append(goal)
    staging_terminal = goals[5]
    measured_terminal = goals[-1].copy()
    measured_terminal[1] += 0.001
    extension_intermediate = goals[1].copy()
    extension_intermediate[1] -= 0.02
    cradle_intermediate = goals[3].copy()
    cradle_intermediate[1] -= 0.02
    compact_intermediate = measured_terminal.copy()
    compact_intermediate[1] -= 0.02
    measured = iter(
        (
            current.copy(),
            current.copy(),
            extension_intermediate,
            goals[1].copy(),
            goals[1].copy(),
            cradle_intermediate,
            goals[3].copy(),
            goals[3].copy(),
            compact_intermediate,
            measured_terminal.copy(),
        )
    )
    node._measured_left_solution = lambda: next(measured)
    node._cancel = threading.Event()
    now = SimpleNamespace(nanoseconds=2_000_000_000)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    node._cached_post_retreat_plan = {
        'requires_gravity_support': False,
        'start': current.copy(),
        'shelf_front_x': 0.95,
        'attached_corners': node._held_book_corners.copy(),
        'legs': [
            (goal.copy(), phase)
            for goal, phase in zip(goals, phases)
        ],
        'staging_terminal': staging_terminal.copy(),
        'terminal': goals[-1].copy(),
        'compact_radius': 0.436,
    }
    node._solve_supported_post_retreat_staging = lambda *args: pytest.fail(
        'cached staging must not be solved again after base retreat'
    )
    node._supported_compact_goals = lambda *args: pytest.fail(
        'cached compact goals must not be regenerated after base retreat'
    )
    post_retreat_checks = []
    node._carried_post_retreat_transition_is_safe = (
        lambda start, end, corners, shelf_front: post_retreat_checks.append(
            (np.asarray(start).copy(), np.asarray(end).copy(), shelf_front)
        ) or True
    )
    support_checks = []
    node._gravity_supported_transition_is_safe = (
        lambda start, end: support_checks.append(
            (np.asarray(start).copy(), np.asarray(end).copy())
        ) or True
    )
    node._carried_navigation_radius = lambda *args: 0.436
    node._plan_carried_joint_route = lambda *args, **kwargs: pytest.fail(
        'cached route must not be replanned while the cradled book is waiting'
    )
    calls = []
    probes = []

    def probe(command, phase, **kwargs):
        calls.append(f'probe:{phase}')
        probes.append((command, phase, kwargs))
        return True

    node._fresh_retention_probe = probe
    retention_checks = []
    node._retention_after_leg = (
        lambda command, phase, leg: retention_checks.append(
            (command, phase, leg)
        ) or True
    )
    prepared_goals = [object(), object(), object()]
    prepared_groups = []

    def prepare(legs):
        group_index = len(prepared_groups)
        calls.append(f'prepare:{group_index}')
        prepared_groups.append(list(legs))
        return prepared_goals[group_index], 10.0 + group_index

    node._make_retained_arm_trajectory_goal = prepare
    node.timeout = 120.0
    node.arm_client = SimpleNamespace(
        wait_for_server=lambda **kwargs: calls.append('server_ready') or True
    )
    sent = []

    def send(goal, duration, legs, command):
        state = (
            bool(getattr(node, '_retention_probe_active', False)),
            bool(getattr(node, '_gravity_supported_payload', False)),
            bool(getattr(node, '_payload_robot_watchdog_enabled', False)),
        )
        group_index = len(sent)
        calls.append(f'send:{group_index}')
        sent.append((goal, duration, list(legs), command, state))
        return True, False

    node._send_retained_arm_trajectory = send
    endpoint_watchdog_states = []

    def endpoint_hazard(**kwargs):
        endpoint_watchdog_states.append(
            (
                bool(getattr(node, '_retention_probe_active', False)),
                bool(getattr(node, '_gravity_supported_payload', False)),
                bool(getattr(node, '_payload_robot_watchdog_enabled', False)),
            )
        )
        return None

    node._payload_hazard_reason = endpoint_hazard
    node._execute_retained_arm_legs = lambda *args, **kwargs: pytest.fail(
        'a cached route must use one continuous controller goal'
    )
    statuses = []
    node._publish_status = (
        lambda event, **fields: statuses.append((event, fields))
    )

    assert node._compact_transport()

    assert [
        phase
        for group in prepared_groups
        for _, _, phase in group
    ] == phases
    assert [[phase for _, _, phase in group] for group in prepared_groups] == [
        ['post_retreat_clearance_extension'] * 2,
        ['post_retreat_cradle_roll'] * 2,
        [
            'supported_cradle_lowering',
            'supported_cradle_retraction',
            'compact_transport',
            'compact_transport',
        ],
    ]
    assert [duration for _, duration, _ in prepared_groups[1]] == pytest.approx(
        [0.375, 0.375]
    )
    assert len(post_retreat_checks) == 3
    assert len(support_checks) == 1
    np.testing.assert_allclose(post_retreat_checks[0][0], current)
    np.testing.assert_allclose(post_retreat_checks[0][1], current)
    np.testing.assert_allclose(post_retreat_checks[1][0], current)
    np.testing.assert_allclose(post_retreat_checks[1][1], goals[0])
    np.testing.assert_allclose(post_retreat_checks[2][0], measured_terminal)
    np.testing.assert_allclose(post_retreat_checks[2][1], measured_terminal)
    assert len(sent) == 3
    for index, (sent_goal, duration, sent_legs, command, _) in enumerate(sent):
        assert sent_goal is prepared_goals[index]
        assert duration == pytest.approx(10.0 + index)
        assert [leg[1:] for leg in sent_legs] == [
            leg[1:] for leg in prepared_groups[index]
        ]
        for sent_leg, prepared_leg in zip(sent_legs, prepared_groups[index]):
            np.testing.assert_allclose(sent_leg[0], prepared_leg[0])
        assert command == 'compact_transport'
    assert [entry[4] for entry in sent] == [
        (False, False, False),
        (True, True, True),
        (False, True, False),
    ]
    assert (True, True, True) in endpoint_watchdog_states
    assert endpoint_watchdog_states.count((True, True, True)) >= 3
    assert calls == [
        'prepare:0',
        'prepare:1',
        'prepare:2',
        'server_ready',
        'probe:post_retreat',
        'send:0',
        'probe:post_retreat_clearance_extension',
        'send:1',
        'probe:post_retreat_cradle_roll',
        'send:2',
        'probe:compact_transport_final',
    ]
    assert probes == [
        (
            'compact_transport',
            'post_retreat',
            {'require_new_sample': False},
        ),
        (
            'compact_transport',
            'post_retreat_clearance_extension',
            {'leg': 1},
        ),
        (
            'compact_transport',
            'post_retreat_cradle_roll',
            {'leg': 3},
        ),
        (
            'compact_transport',
            'compact_transport_final',
            {'leg': len(phases)},
        ),
    ]
    assert retention_checks == [
        ('compact_transport', 'post_retreat_cradle_roll', 3)
    ]
    np.testing.assert_allclose(
        node._carried_staging_solution,
        staging_terminal,
    )
    assert node._post_retreat_shelf_front_x is None
    assert not node._supported_post_retreat_staging_required
    assert node._cached_post_retreat_plan is None
    assert node._gravity_supported_payload
    assert not node._retention_probe_active
    assert not node._payload_robot_watchdog_enabled
    assert statuses[-1] == (
        'transport_compact',
        {
            'command': 'compact_transport',
            'waypoints': len(phases),
            'planar_radius': pytest.approx(0.436),
        },
    )


def test_deferred_compaction_q7_settle_hazard_blocks_supported_group(monkeypatch):
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._post_retreat_shelf_front_x = 0.95
    node._gravity_supported_payload = False
    node._supported_post_retreat_staging_required = True
    node._retention_probe_active = False
    node._payload_robot_watchdog_enabled = False
    node.carried_navigation_radius_limit = 0.45
    node.carried_cradle_transfer = 0.75
    node.grasp_contact_max_age = 0.75
    node.timeout = 1.0
    node._cancel = threading.Event()
    current = np.zeros(8)
    node._cached_post_retreat_plan = _minimal_post_retreat_cache(node, current)
    goals = [
        np.asarray(solution, dtype=float)
        for solution, _ in node._cached_post_retreat_plan['legs']
    ]
    q7_intermediate = goals[1].copy()
    q7_intermediate[1] -= 0.02
    measured = iter(
        (
            current.copy(),
            current.copy(),
            goals[0].copy(),
            goals[0].copy(),
            q7_intermediate,
        )
    )
    node._measured_left_solution = lambda: next(measured)
    node._carried_post_retreat_transition_is_safe = lambda *args: True
    node._gravity_supported_transition_is_safe = lambda *args: True
    node._make_retained_arm_trajectory_goal = lambda legs: (object(), 1.0)
    node.arm_client = SimpleNamespace(wait_for_server=lambda **kwargs: True)
    sent_groups = []

    def send(goal, duration, legs, command):
        sent_groups.append([phase for _, _, phase in legs])
        return True, False

    node._send_retained_arm_trajectory = send
    probes = []
    node._fresh_retention_probe = (
        lambda command, phase, **kwargs: probes.append(phase) or True
    )
    node._retention_after_leg = lambda *args, **kwargs: pytest.fail(
        'a q7 endpoint hazard must stop before endpoint retention checks'
    )
    q7_hazard_checks = []

    def hazard(**kwargs):
        if node._payload_robot_watchdog_enabled:
            q7_hazard_checks.append(
                (
                    node._retention_probe_active,
                    node._gravity_supported_payload,
                    node._payload_robot_watchdog_enabled,
                )
            )
            if len(q7_hazard_checks) >= 2:
                return 'payload_robot_contact'
        return None

    node._payload_hazard_reason = hazard
    clock_ns = [0]
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=clock_ns[0])
    )
    monkeypatch.setattr(
        manipulation_node.time,
        'sleep',
        lambda duration: clock_ns.__setitem__(
            0, clock_ns[0] + int(duration * 1e9)
        ),
    )
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    with pytest.raises(
        RuntimeError,
        match='post-retreat cradle roll missed its terminal state',
    ):
        node._compact_transport()

    assert sent_groups == [
        ['post_retreat_clearance_extension'],
        ['post_retreat_cradle_roll'],
    ]
    assert probes == [
        'post_retreat',
        'post_retreat_clearance_extension',
    ]
    assert q7_hazard_checks == [(True, True, True), (True, True, True)]
    assert not node._retention_probe_active
    assert not node._payload_robot_watchdog_enabled
    assert node._gravity_supported_payload
    assert any(
        event == 'grasp_lost'
        and fields.get('reason') == 'payload_robot_contact'
        and fields.get('endpoint_settle') is True
        for event, fields in statuses
    )


def test_deferred_compaction_rejects_a_premature_supported_runtime_state():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._post_retreat_shelf_front_x = 0.95
    node._gravity_supported_payload = True
    node._supported_post_retreat_staging_required = True
    node._cached_post_retreat_plan = {
        'requires_gravity_support': False,
        'sentinel': 'must-not-be-used',
    }
    node._measured_left_solution = lambda: np.zeros(8)

    with pytest.raises(RuntimeError, match='vertical bilateral pinch'):
        node._compact_transport()


def test_deferred_compaction_rejects_a_missing_cache_without_opening():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._post_retreat_shelf_front_x = 0.95
    node._gravity_supported_payload = False
    node._supported_post_retreat_staging_required = True
    node._cached_post_retreat_plan = None
    node.carried_navigation_radius_limit = 0.45
    node._fresh_retention_probe = lambda *args, **kwargs: True
    node._measured_left_solution = lambda: np.zeros(8)
    node._carried_post_retreat_transition_is_safe = lambda *args: True
    node._gravity_supported_transition_is_safe = lambda *args: True
    node._carried_navigation_radius = lambda *args: 0.70
    node._plan_carried_joint_route = lambda *args, **kwargs: pytest.fail(
        'missing cache must not trigger a slow post-retreat replan'
    )
    opened = []
    node._open_gripper = lambda: opened.append(True) or True

    with pytest.raises(RuntimeError, match='cached post-retreat plan is unavailable'):
        node._compact_transport()

    assert opened == []
    assert node._held_book_corners is not None


def test_deferred_compaction_rejects_torso_only_cache_mismatch_without_motion():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._post_retreat_shelf_front_x = 0.95
    node._gravity_supported_payload = False
    node._supported_post_retreat_staging_required = True
    node.carried_navigation_radius_limit = 0.45
    cached_start = np.zeros(8)
    current = cached_start.copy()
    current[0] = 0.02
    cached_goal = np.ones(8)
    cached_goal[0] = cached_start[0]
    node._cached_post_retreat_plan = {
        'requires_gravity_support': False,
        'start': cached_start,
        'shelf_front_x': 0.95,
        'attached_corners': node._held_book_corners.copy(),
        'legs': [(cached_goal.copy(), 'supported_cradle_lowering')],
        'staging_terminal': cached_goal.copy(),
        'terminal': cached_goal.copy(),
        'compact_radius': 0.436,
    }
    node._fresh_retention_probe = lambda *args, **kwargs: True
    node._measured_left_solution = lambda: current.copy()
    node._carried_post_retreat_transition_is_safe = lambda *args: True
    node._gravity_supported_transition_is_safe = lambda *args: True
    node._carried_navigation_radius = lambda *args: 0.70
    node._execute_retained_arm_legs = lambda *args, **kwargs: pytest.fail(
        'a torso-mismatched cache must not dispatch arm motion'
    )
    opened = []
    node._open_gripper = lambda: opened.append(True) or True

    with pytest.raises(
        RuntimeError,
        match='cached post-retreat plan does not match measured state',
    ):
        node._compact_transport()

    assert opened == []
    assert node._held_book_corners is not None


def test_deferred_compaction_rejects_a_cached_arm_leg_that_changes_torso():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._post_retreat_shelf_front_x = 0.95
    node._gravity_supported_payload = False
    node._supported_post_retreat_staging_required = True
    node.carried_navigation_radius_limit = 0.45
    current = np.zeros(8)
    invalid_goal = np.ones(8)
    valid_terminal = invalid_goal.copy()
    valid_terminal[0] = current[0]
    node._cached_post_retreat_plan = {
        'requires_gravity_support': False,
        'start': current.copy(),
        'shelf_front_x': 0.95,
        'attached_corners': node._held_book_corners.copy(),
        'legs': [(invalid_goal.copy(), 'supported_cradle_lowering')],
        'staging_terminal': valid_terminal.copy(),
        'terminal': valid_terminal.copy(),
        'compact_radius': 0.436,
    }
    node._measured_left_solution = lambda: current.copy()
    node._make_retained_arm_trajectory_goal = lambda *args: pytest.fail(
        'a torso-changing cached leg must not build or dispatch an arm goal'
    )

    with pytest.raises(RuntimeError, match='cached post-retreat plan'):
        node._compact_transport()


def test_deferred_compaction_rejects_unsafe_identity_matching_measured_start():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._post_retreat_shelf_front_x = 0.95
    node._gravity_supported_payload = False
    node._supported_post_retreat_staging_required = True
    node.carried_navigation_radius_limit = 0.45
    node.carried_cradle_transfer = 0.75
    current = np.zeros(8)
    node._cached_post_retreat_plan = _minimal_post_retreat_cache(node, current)
    node._measured_left_solution = lambda: current.copy()
    node._carried_post_retreat_transition_is_safe = lambda *args: False
    node._gravity_supported_transition_is_safe = lambda *args: True
    node._make_retained_arm_trajectory_goal = lambda *args: pytest.fail(
        'an unsafe measured start must not build or dispatch an arm goal'
    )

    with pytest.raises(RuntimeError, match='measured cached start'):
        node._compact_transport()


def test_compact_transport_retains_and_halts_a_lost_payload():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._post_retreat_shelf_front_x = 0.95
    node._gravity_supported_payload = False
    node._supported_post_retreat_staging_required = True
    node.carried_navigation_radius_limit = 0.45
    current = np.zeros(8)
    node._measured_left_solution = lambda: current.copy()
    node._cached_post_retreat_plan = _minimal_post_retreat_cache(node, current)
    node.carried_cradle_transfer = 0.75
    node.timeout = 120.0
    node.arm_client = SimpleNamespace(wait_for_server=lambda **kwargs: True)
    node._carried_post_retreat_transition_is_safe = lambda *args: True
    node._gravity_supported_transition_is_safe = lambda *args: True
    node._make_retained_arm_trajectory_goal = lambda legs: (object(), 1.0)
    node._fresh_retention_probe = lambda *args, **kwargs: False
    node._send_retained_arm_trajectory = lambda *args, **kwargs: pytest.fail(
        'failed post-retreat contact must prevent arm dispatch'
    )
    released = []

    def release():
        released.append(True)
        node._held_book_corners = None
        return True

    node._open_gripper = release

    with pytest.raises(
        RuntimeError,
        match='target-book contact was not retained before compaction',
    ):
        node._compact_transport()

    assert released == []
    assert node._held_book_corners is not None


def test_compact_transport_rechecks_fresh_contact_immediately_before_motion():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._post_retreat_shelf_front_x = None
    node.carried_navigation_radius_limit = 0.45
    current = np.zeros(8)
    compact_goal = manipulation_node.CARRY.copy()
    compact_goal[0] = current[0]
    node._measured_left_solution = lambda: current.copy()
    probe_results = iter((True, False))
    probes = []

    def fresh_probe(command, phase, **kwargs):
        probes.append((command, phase))
        return next(probe_results)

    node._fresh_retention_probe = fresh_probe
    node._carried_robot_transition_is_safe = lambda *args: True
    node._carried_navigation_radius = lambda solution, corners: (
        0.70 if np.allclose(solution, current) else 0.436
    )
    node._plan_carried_joint_route = (
        lambda *args, **kwargs: [compact_goal.copy()]
    )
    moves = []
    node._move_arm_solution = lambda *args: moves.append(args) or True
    opened = []

    def open_gripper():
        opened.append(True)
        node._held_book_corners = None
        return True

    node._open_gripper = open_gripper

    with pytest.raises(RuntimeError, match='compact_transport_failed'):
        node._compact_transport()

    assert probes == [
        ('compact_transport', 'post_retreat'),
        ('compact_transport', 'pre_compact_motion'),
    ]
    assert moves == []
    assert opened == []
    assert node._held_book_corners is not None


def test_place_fails_closed_before_arm_motion_without_fresh_navigation_contact():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._carried_staging_solution = manipulation_node.CARRY.copy()
    probes = []
    node._fresh_retention_probe = (
        lambda command, phase: probes.append((command, phase)) or False
    )
    opened = []

    def open_gripper():
        opened.append(True)
        node._held_book_corners = None
        return True

    node._open_gripper = open_gripper
    node._wait_for_perception_point = lambda *args: (_ for _ in ()).throw(
        AssertionError('place planning must not start after a failed fresh gate')
    )

    with pytest.raises(
        RuntimeError,
        match='target-book contact was not retained before placement',
    ):
        node._place()

    assert probes == [('place', 'post_navigation')]
    assert opened == []
    assert node._held_book_corners is not None


def test_place_rechecks_fresh_contact_after_planning_before_motion():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._carried_staging_solution = manipulation_node.CARRY.copy()
    node.place_clearance = 0.08
    node.cartesian_clearance = 0.45
    node.cartesian_step = 0.06
    node.place_torso_height = 0.25
    node.place_joint_limit_margin = 0.03
    node._measured_left_solution = lambda: manipulation_node.CARRY.copy()
    node._wait_for_perception_point = lambda attribute: np.asarray(
        [0.70, 0.0, 1.0]
    )
    probe_results = iter((True, False))
    probes = []

    def fresh_probe(command, phase, **kwargs):
        probes.append((command, phase))
        return next(probe_results)

    node._fresh_retention_probe = fresh_probe
    node._carried_robot_transition_is_safe = lambda *args: True

    def plan_carried_route(start, goals, corners, **kwargs):
        return [np.asarray(goals[-1], dtype=float)] if goals else []

    node._plan_carried_joint_route = plan_carried_route

    def solve_cartesian_path(positions, rotations, torso_height, **kwargs):
        solution = manipulation_node.CARRY.copy()
        solution[0] = torso_height
        assert kwargs['candidate_validator']([solution], [])
        return [solution], 0, 1.0, []

    node._solve_cartesian_path = solve_cartesian_path
    node._plan_retracted_transition = lambda *args: []
    node._publish_status = lambda *args, **kwargs: None
    torso_moves = []
    node._move_torso = lambda *args, **kwargs: torso_moves.append(args) or True
    opened = []

    def open_gripper():
        opened.append(True)
        node._held_book_corners = None
        return True

    node._open_gripper = open_gripper

    with pytest.raises(
        RuntimeError,
        match='target-book contact was not retained before placement',
    ):
        node._place()

    assert probes == [
        ('place', 'post_navigation'),
        ('place', 'pre_place_motion'),
    ]
    assert torso_moves == []
    assert opened == []
    assert node._held_book_corners is not None


def test_gravity_supported_place_dispatches_the_validated_supported_route():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._gravity_supported_payload = True
    node.place_clearance = 0.08
    node.cartesian_clearance = 0.45
    node.cartesian_step = 0.06
    node.place_torso_height = 0.30
    node.place_joint_limit_margin = 0.03
    carried_start = manipulation_node.SUPPORTED_CARRY.copy()
    node._measured_left_solution = lambda: carried_start.copy()
    place_staging = np.arange(8, dtype=float)
    node._carried_staging_solution = place_staging.copy()
    node._wait_for_perception_point = lambda attribute: np.asarray(
        [0.72, -0.068, 0.752]
    )
    probes = []
    node._fresh_retention_probe = lambda command, phase, **kwargs: (
        probes.append((command, phase, kwargs)) or True
    )
    support_midpoint = np.full(8, 4.0)
    node._supported_compact_goals = lambda start: [
        support_midpoint.copy(),
        carried_start.copy(),
    ]
    node._carried_robot_transition_is_safe = lambda *args: True
    node._gravity_supported_transition_is_safe = lambda *args: True
    planned = []

    def plan_carried_route(start, goals, corners, **kwargs):
        route = [np.asarray(goal, dtype=float).copy() for goal in goals]
        planned.append((np.asarray(start).copy(), route, kwargs))
        return route

    node._plan_carried_joint_route = plan_carried_route
    solve_calls = []
    bridge = np.full(8, 5.0)
    clearance_solution = np.full(8, 6.0)
    release_solution = np.full(8, 7.0)

    def solve_cartesian_path(positions, rotations, torso_height, **kwargs):
        solve_calls.append((rotations, torso_height, kwargs))
        assert kwargs['candidate_validator'](
            [clearance_solution, release_solution],
            [bridge],
        )
        return [clearance_solution, release_solution], 0, 1.0, [bridge]

    node._solve_cartesian_path = solve_cartesian_path
    node._plan_retracted_transition = lambda *args: []
    node._publish_status = lambda *args, **kwargs: None
    node._move_torso = lambda *args, **kwargs: True
    dispatched = []

    def execute(legs, command):
        dispatched.extend(legs)
        return True, len(legs), False

    node._execute_retained_arm_legs = execute
    node._open_gripper = lambda: True
    node._return_from_bin = lambda *args: True

    assert node._place()

    assert len(planned) == 2
    assert all(
        call_kwargs['require_gravity_support'] is True
        for _, _, call_kwargs in planned
    )
    np.testing.assert_allclose(planned[0][1][0], support_midpoint)
    expected_place_staging = place_staging.copy()
    expected_place_staging[0] = node.place_torso_height
    np.testing.assert_allclose(planned[0][1][1], expected_place_staging)
    np.testing.assert_allclose(planned[1][1][0], bridge)
    np.testing.assert_allclose(planned[1][1][1], clearance_solution)
    assert solve_calls[0][1] == pytest.approx(0.30)
    assert solve_calls[0][2]['joint_limit_margin'] == pytest.approx(0.10)
    assert len(solve_calls[0][0]) == len(
        manipulation_node.supported_bin_place_orientations()
    )
    assert [phase for _, _, phase in dispatched] == [
        'bin_transition',
        'bin_transition',
        'bin_transition',
        'bin_clearance',
        'bin_approach',
    ]
    assert [phase for _, phase, _ in probes] == [
        'post_navigation',
        'pre_place_motion',
        'torso',
    ]


def test_compact_transport_rejects_unsafe_measured_current_pose_before_radius():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node.carried_navigation_radius_limit = 0.45
    current = manipulation_node.CARRY.copy()
    node._measured_left_solution = lambda: current.copy()
    node._fresh_retention_probe = lambda *args, **kwargs: True
    node._target_contact_recent = lambda **kwargs: True
    node._carried_robot_transition_is_safe = lambda *args: False
    node._carried_navigation_radius = lambda *args: (_ for _ in ()).throw(
        AssertionError('radius must not be accepted before collision validation')
    )

    with pytest.raises(RuntimeError, match='Current carried pose is not payload-safe'):
        node._compact_transport()


def test_compact_transport_rejects_unsafe_measured_terminal_pose():
    node = object.__new__(manipulation_node.ManipulationNode)
    _stub_retained_leg_dispatch(node)
    node._held_book_corners = np.zeros((8, 3))
    node.carried_navigation_radius_limit = 0.45
    current = np.zeros(8)
    terminal = manipulation_node.CARRY.copy()
    terminal[0] = current[0]
    measured = iter((current.copy(), terminal.copy()))
    node._measured_left_solution = lambda: next(measured)
    node._fresh_retention_probe = lambda *args, **kwargs: True
    node._target_contact_recent = lambda **kwargs: True
    validations = []

    def pose_is_safe(start, end, corners):
        validations.append(np.asarray(start).copy())
        return len(validations) == 1

    node._carried_robot_transition_is_safe = pose_is_safe
    node._carried_navigation_radius = lambda solution, corners: (
        0.70 if np.allclose(solution, current) else 0.4363
    )
    node._plan_carried_joint_route = lambda *args, **kwargs: [terminal.copy()]
    node._move_arm_solution = lambda *args: True
    node._publish_status = lambda *args, **kwargs: None

    with pytest.raises(
        RuntimeError,
        match='Measured compact transport pose is not payload-safe',
    ):
        node._compact_transport()

    assert len(validations) == 2
    np.testing.assert_allclose(validations[0], current)
    np.testing.assert_allclose(validations[1], terminal)


def _force_vector(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def _joint_wrench(body_1_force, body_2_force):
    return SimpleNamespace(
        body_1_wrench=SimpleNamespace(force=body_1_force),
        body_2_wrench=SimpleNamespace(force=body_2_force),
    )


def test_message_stamp_prefers_producer_time_and_rejects_invalid_fields():
    stamped = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=2, nanosec=345),
        )
    )
    invalid = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=-1, nanosec=0),
        )
    )
    same_step_future = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=2, nanosec=50_000_000),
        )
    )

    assert manipulation_node._message_stamp_ns(
        stamped,
        999,
    ) == 2_000_000_345
    assert manipulation_node._message_stamp_ns(invalid, 999) == 999
    assert manipulation_node._message_stamp_ns(
        same_step_future,
        2_000_000_000,
    ) == 2_000_000_000


def test_joint_state_callback_records_gripper_velocity_and_effort():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.joints = {}
    node._gripper_feedback_samples = manipulation_node.deque(maxlen=128)
    now = SimpleNamespace(nanoseconds=456_000_000)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    message = SimpleNamespace(
        name=['arm_left_1_joint', 'gripper_left_finger_joint'],
        position=[0.4, 0.037],
        velocity=[0.0, -0.001],
        effort=[0.1, 0.42],
    )

    node._on_joint_state(message)

    assert node.joints['gripper_left_finger_joint'] == pytest.approx(0.037)
    assert tuple(node._gripper_feedback_samples) == (
        manipulation_node.GripperFeedback(
            now.nanoseconds,
            0.037,
            -0.001,
            0.42,
        ),
    )


def test_contact_force_uses_gripper_body_and_sums_contact_points():
    contact = SimpleNamespace(
        wrenches=[
            _joint_wrench(
                _force_vector(3.0, 4.0, 0.0),
                _force_vector(30.0, 40.0, 0.0),
            ),
            _joint_wrench(
                _force_vector(0.0, 0.0, 12.0),
                _force_vector(0.0, 0.0, 120.0),
            ),
        ]
    )

    first = manipulation_node._contact_force_magnitude(
        contact,
        gripper_is_collision1=True,
    )
    second = manipulation_node._contact_force_magnitude(
        contact,
        gripper_is_collision1=False,
    )

    assert first == pytest.approx(17.0)
    assert second == pytest.approx(170.0)


@pytest.mark.parametrize(
    'contact',
    (
        SimpleNamespace(wrenches=[]),
        SimpleNamespace(
            wrenches=[
                _joint_wrench(
                    _force_vector(math.nan, 0.0, 0.0),
                    _force_vector(0.0, 0.0, 0.0),
                )
            ]
        ),
    ),
)
def test_contact_force_rejects_missing_or_nonfinite_wrenches(contact):
    assert manipulation_node._contact_force_magnitude(
        contact,
        gripper_is_collision1=True,
    ) is None


def test_contact_callback_tracks_left_and_right_target_fingers_separately():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.target_colour = 'red'
    node._left_target_contact_ns = 0
    node._right_target_contact_ns = 0
    now = SimpleNamespace(nanoseconds=123456789)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    book = SimpleNamespace(name='book_col_red::base_link::book_collision')
    left = SimpleNamespace(
        name='tiago::gripper_left_fingertip_left_collision'
    )
    right = SimpleNamespace(
        name='tiago::gripper_left_fingertip_right_collision'
    )
    message = SimpleNamespace(
        contacts=[
            SimpleNamespace(collision1=left, collision2=book),
            SimpleNamespace(collision1=right, collision2=book),
        ]
    )

    node._on_contacts(message)

    assert node._left_target_contact_ns == now.nanoseconds
    assert node._right_target_contact_ns == now.nanoseconds


def test_contact_callback_records_target_specific_finger_force_histories():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.target_colour = 'red'
    node.grasp_contact_max_age = 0.75
    node._target_book_model = 'book_col_3_row_2_red'
    node._book_contact_samples = {}
    node._book_contact_force_samples = {}
    node._left_target_contact_ns = 0
    node._right_target_contact_ns = 0
    node._contact_epoch = 0
    node._contact_generation = 0
    node._adaptive_close_active = True
    node._adaptive_overload_latched = None
    node.adaptive_contact_force_maximum = 1.0
    now = SimpleNamespace(nanoseconds=123_000_000)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    book = SimpleNamespace(
        name=(
            'book_col_3_row_2_red::book_base_link::'
            'base_link_book_collision'
        )
    )
    left = SimpleNamespace(
        name='tiago_pro::gripper_left_fingertip_left_link::collision'
    )
    right = SimpleNamespace(
        name='tiago_pro::gripper_left_fingertip_right_link::collision'
    )
    message = SimpleNamespace(
        contacts=[
            SimpleNamespace(
                collision1=left,
                collision2=book,
                wrenches=[
                    _joint_wrench(
                        _force_vector(0.3, 0.4, 0.0),
                        _force_vector(3.0, 4.0, 0.0),
                    )
                ],
            ),
            SimpleNamespace(
                collision1=book,
                collision2=right,
                wrenches=[
                    _joint_wrench(
                        _force_vector(6.0, 8.0, 0.0),
                        _force_vector(0.0, 0.0, 1.25),
                    )
                ],
            ),
        ]
    )

    node._on_contacts(message)
    # Repeated delivery of the same collision pair and stamp must form one
    # force frame without adding duplicate force or temporal samples.
    node._on_contacts(message)

    left_history, right_history = node._book_contact_force_samples[
        'book_col_3_row_2_red'
    ]
    assert tuple(left_history) == (
        manipulation_node.ForceSample(now.nanoseconds, 0.5),
    )
    assert tuple(right_history) == (
        manipulation_node.ForceSample(now.nanoseconds, 1.25),
    )
    assert node._adaptive_overload_latched == 'force_overload'
    node._clear_target_contact_samples()
    assert node._adaptive_overload_latched == 'force_overload'


def test_contact_callback_latches_one_concrete_book_and_ignores_other_red_books():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.target_colour = 'red'
    node.grasp_contact_max_age = 0.75
    node._target_book_model = None
    node._book_contact_samples = {}
    node._left_target_contact_ns = 0
    node._right_target_contact_ns = 0
    node._target_robot_contact_latched = False
    node._contact_generation = 0
    now = SimpleNamespace(nanoseconds=123456789)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )
    selected = SimpleNamespace(
        name=(
            'book_col_2_row_1_red::book_base_link::'
            'base_link_book_collision'
        )
    )
    other = SimpleNamespace(
        name=(
            'book_col_3_row_2_red::book_base_link::'
            'base_link_book_collision'
        )
    )
    left = SimpleNamespace(
        name='tiago_pro::gripper_left_fingertip_left_link::collision'
    )
    right = SimpleNamespace(
        name='tiago_pro::gripper_left_fingertip_right_link::collision'
    )
    arm = SimpleNamespace(
        name='tiago_pro::arm_left_4_link::arm_left_4_link_collision'
    )

    node._on_contacts(
        SimpleNamespace(
            contacts=[
                SimpleNamespace(collision1=left, collision2=selected),
                SimpleNamespace(collision1=right, collision2=selected),
            ]
        )
    )

    assert node._target_book_model == 'book_col_2_row_1_red'
    assert statuses == [
        (
            'target_book_latched',
            {'model': 'book_col_2_row_1_red'},
        )
    ]

    node._on_contacts(
        SimpleNamespace(
            contacts=[SimpleNamespace(collision1=other, collision2=arm)]
        )
    )
    assert not node._target_robot_contact_latched

    node._on_contacts(
        SimpleNamespace(
            contacts=[SimpleNamespace(collision1=selected, collision2=arm)]
        )
    )
    assert node._target_robot_contact_latched


@pytest.mark.parametrize(
    ('robot_book_name', 'expected_hazard'),
    (
        ('book_col_2_row_1_red', True),
        ('book_col_3_row_2_red', False),
    ),
)
def test_contact_callback_resolves_same_message_robot_contact_after_latching(
    robot_book_name,
    expected_hazard,
):
    node = object.__new__(manipulation_node.ManipulationNode)
    node.target_colour = 'red'
    node.grasp_contact_max_age = 0.75
    node._lock = threading.Lock()
    node._contact_epoch = 0
    node._contact_generation = 0
    node._target_book_model = None
    node._book_contact_samples = {}
    node._left_target_contact_ns = 0
    node._right_target_contact_ns = 0
    node._target_robot_contact_latched = False
    now = SimpleNamespace(nanoseconds=123456789)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )
    selected = SimpleNamespace(
        name=(
            'book_col_2_row_1_red::book_base_link::'
            'base_link_book_collision'
        )
    )
    robot_book = SimpleNamespace(
        name=(
            f'{robot_book_name}::book_base_link::'
            'base_link_book_collision'
        )
    )
    left = SimpleNamespace(
        name='tiago_pro::gripper_left_fingertip_left_link::collision'
    )
    right = SimpleNamespace(
        name='tiago_pro::gripper_left_fingertip_right_link::collision'
    )
    arm = SimpleNamespace(
        name='tiago_pro::arm_left_4_link::arm_left_4_link_collision'
    )

    node._on_contacts(
        SimpleNamespace(
            contacts=[
                SimpleNamespace(collision1=left, collision2=selected),
                SimpleNamespace(collision1=right, collision2=selected),
                SimpleNamespace(collision1=robot_book, collision2=arm),
            ]
        )
    )

    assert node._target_book_model == 'book_col_2_row_1_red'
    assert node._target_robot_contact_latched is expected_hazard
    assert statuses == [
        (
            'target_book_latched',
            {'model': 'book_col_2_row_1_red'},
        )
    ]


def test_contact_callback_fails_closed_for_anonymous_held_book_robot_contact():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.target_colour = 'red'
    node.grasp_contact_max_age = 0.75
    node._contact_epoch = 0
    node._contact_generation = 0
    node._target_book_model = None
    node._book_contact_samples = {}
    node._left_target_contact_ns = 0
    node._right_target_contact_ns = 0
    node._target_robot_contact_latched = False
    now = SimpleNamespace(nanoseconds=123456789)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    book = SimpleNamespace(name='book_base_link::base_link_book_collision')
    left = SimpleNamespace(
        name='tiago_pro::gripper_left_fingertip_left_link::collision'
    )
    right = SimpleNamespace(
        name='tiago_pro::gripper_left_fingertip_right_link::collision'
    )
    arm = SimpleNamespace(
        name='tiago_pro::arm_left_4_link::arm_left_4_link_collision'
    )

    node._on_contacts(
        SimpleNamespace(
            contacts=[
                SimpleNamespace(collision1=left, collision2=book),
                SimpleNamespace(collision1=right, collision2=book),
                SimpleNamespace(collision1=book, collision2=arm),
            ]
        )
    )

    assert node._target_book_model is None
    assert node._left_target_contact_ns == now.nanoseconds
    assert node._right_target_contact_ns == now.nanoseconds
    assert node._target_robot_contact_latched


def test_contact_callback_discards_samples_parsed_before_a_contact_epoch_reset(
    monkeypatch,
):
    node = object.__new__(manipulation_node.ManipulationNode)
    node.target_colour = 'red'
    node.grasp_contact_max_age = 0.75
    node._lock = threading.Lock()
    node._contact_epoch = 0
    node._contact_generation = 0
    node._target_book_model = 'book_col_2_row_1_red'
    node._book_contact_samples = {
        'book_col_2_row_1_red': [11, 22],
    }
    node._left_target_contact_ns = 11
    node._right_target_contact_ns = 22
    node._target_robot_contact_latched = True
    now = SimpleNamespace(nanoseconds=123456789)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )
    reset_done = False

    def reset_while_parsing(*args, **kwargs):
        nonlocal reset_done
        if not reset_done:
            reset_done = True
            node._clear_target_contact_samples(
                reset_robot_contact=True,
                reset_target_model=True,
            )
        return False

    monkeypatch.setattr(
        manipulation_node,
        'is_target_book_non_gripper_robot_contact',
        reset_while_parsing,
    )
    book = SimpleNamespace(
        name=(
            'book_col_2_row_1_red::book_base_link::'
            'base_link_book_collision'
        )
    )
    left = SimpleNamespace(
        name='tiago_pro::gripper_left_fingertip_left_link::collision'
    )
    right = SimpleNamespace(
        name='tiago_pro::gripper_left_fingertip_right_link::collision'
    )

    node._on_contacts(
        SimpleNamespace(
            contacts=[
                SimpleNamespace(collision1=left, collision2=book),
                SimpleNamespace(collision1=right, collision2=book),
            ]
        )
    )

    assert node._contact_epoch == 1
    assert node._target_book_model is None
    assert node._book_contact_samples == {}
    assert node._left_target_contact_ns == 0
    assert node._right_target_contact_ns == 0
    assert not node._target_robot_contact_latched
    assert statuses == []


def test_contact_callback_keeps_held_robot_collision_across_sample_epoch_reset(
    monkeypatch,
):
    node = object.__new__(manipulation_node.ManipulationNode)
    node.target_colour = 'red'
    node.grasp_contact_max_age = 0.75
    node._lock = threading.Lock()
    node._contact_epoch = 0
    node._contact_generation = 0
    node._target_book_model = 'book_col_2_row_1_red'
    node._book_contact_samples = {
        'book_col_2_row_1_red': [11, 22],
    }
    node._left_target_contact_ns = 11
    node._right_target_contact_ns = 22
    node._target_robot_contact_latched = False
    node._held_book_corners = np.zeros((8, 3))
    node._transport_lock_engaged = True
    now = SimpleNamespace(nanoseconds=123456789)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    reset_done = False

    def reset_samples_while_parsing(*args, **kwargs):
        nonlocal reset_done
        if not reset_done:
            reset_done = True
            # A fresh-retention probe clears only the sample epoch.  Collision
            # evidence for the still-held target must survive that race.
            node._clear_target_contact_samples()
        return True

    monkeypatch.setattr(
        manipulation_node,
        'is_target_book_non_gripper_robot_contact',
        reset_samples_while_parsing,
    )
    book = SimpleNamespace(
        name=(
            'book_col_2_row_1_red::book_base_link::'
            'base_link_book_collision'
        )
    )
    arm = SimpleNamespace(
        name='tiago_pro::arm_left_4_link::arm_left_4_link_collision'
    )

    node._on_contacts(
        SimpleNamespace(
            contacts=[SimpleNamespace(collision1=book, collision2=arm)]
        )
    )

    assert node._contact_epoch == 1
    assert node._contact_generation == 2
    assert node._target_book_model == 'book_col_2_row_1_red'
    assert node._book_contact_samples == {}
    assert node._left_target_contact_ns == 0
    assert node._right_target_contact_ns == 0
    assert node._target_robot_contact_latched


def test_contact_callback_latches_target_book_robot_collision():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.target_colour = 'red'
    node._left_target_contact_ns = 0
    node._right_target_contact_ns = 0
    node._target_robot_contact_latched = False
    node._target_book_model = 'book_col_3_row_2_red'
    now = SimpleNamespace(nanoseconds=123456789)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    book = SimpleNamespace(
        name='book_col_3_row_2_red::book_base_link::base_link_book_collision'
    )
    arm = SimpleNamespace(
        name='tiago_pro::arm_left_4_link::arm_left_4_link_collision'
    )

    node._on_contacts(
        SimpleNamespace(
            contacts=[SimpleNamespace(collision1=book, collision2=arm)]
        )
    )

    assert node._target_robot_contact_latched
    assert node._left_target_contact_ns == 0
    assert node._right_target_contact_ns == 0


def test_contact_callback_allows_exact_fixed_joint_left_palm_target_contact():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.target_colour = 'red'
    node._left_target_contact_ns = 0
    node._right_target_contact_ns = 0
    node._target_robot_contact_latched = False
    node._target_book_model = 'book_col_3_row_2_red'
    now = SimpleNamespace(nanoseconds=123456789)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    book = SimpleNamespace(
        name='book_col_3_row_2_red::book_base_link::base_link_book_collision'
    )
    palm = SimpleNamespace(
        name=(
            'tiago_pro::arm_left_7_link::'
            'arm_left_7_link_fixed_joint_lump__'
            'gripper_left_base_link_collision_1'
        )
    )

    node._on_contacts(
        SimpleNamespace(
            contacts=[SimpleNamespace(collision1=book, collision2=palm)]
        )
    )

    assert not node._target_robot_contact_latched
    assert node._left_target_contact_ns == 0
    assert node._right_target_contact_ns == 0
    assert getattr(node, '_book_contact_force_samples', {}) == {}


def test_contact_callback_ignores_right_gripper_for_left_arm_grasp_gate():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.target_colour = 'red'
    node._left_target_contact_ns = 0
    node._right_target_contact_ns = 0
    now = SimpleNamespace(nanoseconds=123456789)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    book = SimpleNamespace(name='book_col_red::base_link::book_collision')
    right_gripper_left_finger = SimpleNamespace(
        name='tiago::gripper_right_fingertip_left_collision'
    )
    right_gripper_right_finger = SimpleNamespace(
        name='tiago::gripper_right_fingertip_right_collision'
    )
    message = SimpleNamespace(
        contacts=[
            SimpleNamespace(
                collision1=right_gripper_left_finger,
                collision2=book,
            ),
            SimpleNamespace(
                collision1=right_gripper_right_finger,
                collision2=book,
            ),
        ]
    )

    node._on_contacts(message)

    assert node._left_target_contact_ns == 0
    assert node._right_target_contact_ns == 0


@pytest.mark.parametrize(
    ('settled', 'sample', 'expected'),
    (
        (True, (True, 0.018, True, True, True), True),
        (False, (True, 0.018, True, True, True), False),
        (True, (False, 0.018, True, False, True), False),
    ),
)
def test_fresh_retention_probe_clears_old_samples_before_waiting(
    settled,
    sample,
    expected,
):
    node = object.__new__(manipulation_node.ManipulationNode)
    node.grasp_contact_max_age = 0.75
    node._left_target_contact_ns = 111
    node._right_target_contact_ns = 222
    cleared_before_wait = []

    def wait_sim_duration(duration):
        cleared_before_wait.append(
            (
                node._left_target_contact_ns,
                node._right_target_contact_ns,
                duration,
            )
        )
        return settled

    node._wait_sim_duration = wait_sim_duration
    sampled_ages = []
    node._pinch_sample = lambda **kwargs: (
        sampled_ages.append(kwargs.get('max_age')) or sample
    )
    status = []
    node._publish_status = lambda event, **fields: status.append((event, fields))

    assert node._fresh_retention_probe(
        'pick',
        'mechanical_cradle',
        leg=4,
    ) is expected

    assert cleared_before_wait == [(0, 0, pytest.approx(0.20))]
    assert sampled_ages == [pytest.approx(0.15)]
    assert status[0][0] == 'retention_verified'
    assert status[0][1]['verified'] is expected
    assert status[0][1]['leg'] == 4
    assert [event for event, _ in status[1:]] == (
        [] if expected else ['grasp_lost']
    )


def test_fresh_retention_probe_suppresses_monitor_through_final_sample():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._lock = manipulation_node.threading.Lock()
    node._retention_probe_active = False
    node._left_target_contact_ns = 11
    node._right_target_contact_ns = 22
    node._payload_hazard_latched = None
    node._target_robot_contact_latched = False
    node.grasp_contact_max_age = 0.75
    observed = []

    def wait_sim_duration(duration):
        observed.append(('wait', node._retention_probe_active, duration))
        return True

    def pinch_sample(**kwargs):
        observed.append(
            ('sample', node._retention_probe_active, kwargs['max_age'])
        )
        return True, 0.018, True, True, True

    node._wait_sim_duration = wait_sim_duration
    node._pinch_sample = pinch_sample
    node._publish_status = lambda *args, **kwargs: None

    assert node._fresh_retention_probe('pick', 'mechanical_cradle')

    assert observed == [
        ('wait', True, pytest.approx(0.20)),
        ('sample', True, pytest.approx(0.15)),
    ]
    assert not node._retention_probe_active


@pytest.mark.parametrize('opened', (True, False))
def test_open_gripper_suppresses_payload_monitor_during_intentional_release(
    opened,
):
    node = object.__new__(manipulation_node.ManipulationNode)
    node._lock = manipulation_node.threading.Lock()
    node.gripper_open = 0.069
    attached = np.ones((8, 3))
    node._held_book_corners = attached
    node._carried_staging_solution = np.ones(8)
    node._cached_post_retreat_plan = {'plan': True}
    node._post_retreat_shelf_front_x = 0.95
    node._gravity_supported_payload = True
    node._supported_post_retreat_staging_required = True
    node._transport_lock_engaged = True
    node._payload_hazard_latched = None
    node._payload_monitor_enabled = True
    node._retention_probe_active = False
    node._target_robot_contact_latched = False
    node._left_target_contact_ns = 11
    node._right_target_contact_ns = 22
    node._target_contact_recent = lambda **kwargs: False
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    def command_gripper(position, *, respect_cancel):
        assert respect_cancel is True
        assert position == pytest.approx(node.gripper_open)
        assert not node._payload_monitor_enabled
        assert node._retention_probe_active
        node._monitor_held_payload()
        return opened

    node._command_gripper = command_gripper

    assert node._open_gripper() is opened
    assert statuses == []
    assert not node._retention_probe_active
    if opened:
        assert node._held_book_corners is None
        assert not node._payload_monitor_enabled
        assert node._gripper_open_confirmed
        assert node._left_target_contact_ns == 0
        assert node._right_target_contact_ns == 0
    else:
        assert node._held_book_corners is attached
        assert node._payload_monitor_enabled
        assert not node._gripper_open_confirmed


@pytest.mark.parametrize(
    ('gravity_supported', 'contacts', 'expected'),
    (
        (False, (True, True), True),
        (False, (True, False), False),
        (True, (True, False), True),
        (True, (True, True), True),
        (True, (False, True), False),
        (True, (False, False), False),
    ),
)
def test_retention_gate_switches_from_pinch_to_lower_finger_support(
    gravity_supported,
    contacts,
    expected,
):
    node = object.__new__(manipulation_node.ManipulationNode)
    node._gravity_supported_payload = gravity_supported
    node.grasp_min_position = 0.015
    node.grasp_min_margin = 0.001
    node.grasp_max_position = 0.05
    node.joints = {'gripper_left_finger_joint': 0.0165}
    node._target_contact_sides = lambda **kwargs: contacts

    assert node._target_contact_recent() is expected
    verified, width, left, right, plausible = node._pinch_sample()

    assert verified is expected
    assert width == pytest.approx(0.0165)
    assert (left, right) == contacts
    assert plausible


def test_transport_lock_width_is_valid_only_after_verified_acquisition():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._gravity_supported_payload = False
    node.gripper_closed = 0.0
    node.gripper_transport_lock = 0.010
    node.grasp_min_position = 0.014
    node.grasp_min_margin = 0.001
    node.grasp_max_position = 0.05
    node.joints = {'gripper_left_finger_joint': 0.010001}
    node._target_contact_sides = lambda **kwargs: (True, True)

    node._transport_lock_engaged = False
    unlocked = node._pinch_sample()
    node._transport_lock_engaged = True
    locked = node._pinch_sample()

    assert not unlocked[0]
    assert not unlocked[4]
    assert locked[0]
    assert locked[4]


@pytest.mark.parametrize(
    (
        'contact_samples',
        'width',
        'expected',
        'expected_commands',
        'close_stage',
        'commanded_position',
    ),
    (
        (
            ((True, False), (True, True), (True, True)),
            0.025,
            True,
            [0.02, 0.0165, 0.010],
            'transport_lock',
            0.010,
        ),
        (
            ((True, False),),
            0.025,
            False,
            [0.02, 0.0165],
            'preload',
            0.0165,
        ),
        (
            ((True, True),),
            0.05,
            False,
            [0.02, 0.0165],
            'preload',
            0.0165,
        ),
    ),
)
def test_staged_close_requires_bilateral_contact_and_pinch_band(
    contact_samples,
    width,
    expected,
    expected_commands,
    close_stage,
    commanded_position,
):
    node = object.__new__(manipulation_node.ManipulationNode)
    node.gripper_preclose = 0.02
    node.gripper_preload = 0.0165
    node.gripper_transport_lock = 0.010
    node.gripper_closed = 0.0
    node.grasp_min_position = 0.015
    node.grasp_min_margin = 0.001
    node.grasp_max_position = 0.05
    node.joints = {'gripper_left_finger_joint': width}
    node._left_target_contact_ns = 99
    node._right_target_contact_ns = 88
    commands = []
    status = []
    fresh_probes = []
    node._command_gripper = lambda position: commands.append(position) or True
    node._fresh_retention_probe = lambda command, phase: (
        fresh_probes.append((command, phase)) or True
    )
    samples = iter(contact_samples)
    last_sample = contact_samples[-1]
    node._target_contact_sides = lambda **kwargs: next(samples, last_sample)
    node._publish_status = lambda event, **fields: status.append((event, fields))

    verified, measured, left, right, plausible = node._close_for_grasp()

    assert verified is expected
    assert measured == pytest.approx(width)
    assert (left, right) == last_sample
    assert plausible is (0.016 <= width < 0.05)
    assert commands == expected_commands
    assert fresh_probes == (
        [('pick', 'transport_lock')] if expected else []
    )
    assert node._left_target_contact_ns == 0
    assert node._right_target_contact_ns == 0
    assert status == [
        (
            'gripper_closed',
            {
                'close_stage': close_stage,
                'commanded_position': commanded_position,
                'acquisition_position': width,
                'acquisition_plausible_width': (0.016 <= width < 0.05),
                'transport_lock_engaged': expected,
                'measured_position': width,
                'left_contact': last_sample[0],
                'right_contact': last_sample[1],
                'bilateral_contact': all(last_sample),
                'plausible_width': plausible,
            },
        )
    ]


def test_valid_preclose_applies_transport_lock_and_reverifies_pinch():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.gripper_preclose = 0.02
    node.gripper_preload = 0.0165
    node.gripper_transport_lock = 0.010
    node.gripper_closed = 0.0
    node.grasp_min_position = 0.015
    node.grasp_min_margin = 0.001
    node.grasp_max_position = 0.05
    node._left_target_contact_ns = 99
    node._right_target_contact_ns = 88
    commands = []
    status = []
    samples = iter(
        (
            (True, 0.020, True, True, True),
            (True, 0.018, True, True, True),
            (True, 0.018, True, True, True),
        )
    )
    sample_count = {'value': 0}

    def pinch_sample(**kwargs):
        sample_count['value'] += 1
        return next(samples)

    node._command_gripper = lambda position: commands.append(position) or True
    node._pinch_sample = pinch_sample
    node._fresh_retention_probe = lambda command, phase: (
        command == 'pick' and phase == 'transport_lock'
    )
    node._publish_status = lambda event, **fields: status.append((event, fields))

    verified, measured, left, right, plausible = node._close_for_grasp()

    assert verified
    assert measured == pytest.approx(0.018)
    assert left and right
    assert plausible
    assert sample_count['value'] == 3
    assert commands == [0.020, 0.0165, 0.010]
    assert node.gripper_transport_lock < node.grasp_min_position + node.grasp_min_margin
    assert node.gripper_preload > node.grasp_min_position + node.grasp_min_margin
    assert status == [
        (
            'gripper_closed',
            {
                'close_stage': 'transport_lock',
                'commanded_position': 0.010,
                'acquisition_position': pytest.approx(0.018),
                'acquisition_plausible_width': True,
                'transport_lock_engaged': True,
                'measured_position': pytest.approx(0.018),
                'left_contact': True,
                'right_contact': True,
                'bilateral_contact': True,
                'plausible_width': True,
            },
        )
    ]


@pytest.mark.parametrize(
    'post_hold_sample',
    (
        (False, 0.018, True, False, True),
        (False, 0.0155, True, True, False),
    ),
)
def test_failed_preload_verification_rejects_the_grasp(post_hold_sample):
    node = object.__new__(manipulation_node.ManipulationNode)
    node.gripper_preclose = 0.020
    node.gripper_preload = 0.0165
    node.gripper_transport_lock = 0.010
    node.gripper_closed = 0.0
    node.grasp_min_position = 0.015
    node.grasp_min_margin = 0.001
    node.grasp_max_position = 0.05
    node._left_target_contact_ns = 99
    node._right_target_contact_ns = 88
    commands = []
    status = []
    samples = iter(
        (
            (True, 0.020, True, True, True),
            post_hold_sample,
        )
    )
    node._command_gripper = lambda position: commands.append(position) or True
    node._pinch_sample = lambda: next(samples)
    node._fresh_retention_probe = lambda command, phase: True
    node._publish_status = lambda event, **fields: status.append((event, fields))

    verified, measured, left, right, plausible = node._close_for_grasp()

    assert not verified
    assert (measured, left, right, plausible) == post_hold_sample[1:]
    assert commands == [0.020, 0.0165]
    assert node.gripper_closed not in commands
    assert status[0][0] == 'gripper_closed'
    assert status[0][1]['close_stage'] == 'preload'
    assert status[0][1]['commanded_position'] == pytest.approx(0.0165)
    assert not status[0][1]['transport_lock_engaged']


def test_transport_lock_rejects_stale_command_phase_contacts():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.gripper_preclose = 0.020
    node.gripper_preload = 0.0165
    node.gripper_transport_lock = 0.010
    node.gripper_closed = 0.0
    node.grasp_min_position = 0.014
    node.grasp_min_margin = 0.001
    node.grasp_max_position = 0.05
    node._left_target_contact_ns = 99
    node._right_target_contact_ns = 88
    commands = []
    events = []
    status = []
    samples = iter(
        (
            (True, 0.020, True, True, True),
            (True, 0.0155, True, True, True),
            (True, 0.0100, True, True, True),
        )
    )

    def command_gripper(position):
        commands.append(position)
        events.append(('command', position))
        return True

    def fresh_probe(command, phase):
        events.append(('probe', command, phase))
        return False

    node._command_gripper = command_gripper
    node._pinch_sample = lambda: next(samples)
    node._fresh_retention_probe = fresh_probe
    node._publish_status = lambda event, **fields: status.append((event, fields))

    verified, measured, left, right, plausible = node._close_for_grasp()

    assert not verified
    assert measured == pytest.approx(0.0100)
    assert left and right and plausible
    assert commands == [0.020, 0.0165, 0.010]
    assert events[-2:] == [
        ('command', 0.010),
        ('probe', 'pick', 'transport_lock'),
    ]
    assert not node._transport_lock_engaged
    assert status[0][0] == 'gripper_closed'
    assert not status[0][1]['transport_lock_engaged']


@pytest.mark.parametrize(
    'transport_lock_sample',
    (
        (False, 0.018, True, False, True),
        (False, 0.0149, True, True, False),
    ),
)
def test_failed_transport_lock_verification_rejects_the_grasp(
    transport_lock_sample,
):
    node = object.__new__(manipulation_node.ManipulationNode)
    node.gripper_preclose = 0.020
    node.gripper_preload = 0.0165
    node.gripper_transport_lock = 0.010
    node.gripper_closed = 0.0
    node.grasp_min_position = 0.014
    node.grasp_min_margin = 0.001
    node.grasp_max_position = 0.05
    node._left_target_contact_ns = 99
    node._right_target_contact_ns = 88
    commands = []
    status = []
    samples = iter(
        (
            (True, 0.020, True, True, True),
            (True, 0.0155, True, True, True),
            transport_lock_sample,
        )
    )
    node._command_gripper = lambda position: commands.append(position) or True
    node._pinch_sample = lambda: next(samples)
    node._fresh_retention_probe = lambda command, phase: True
    node._publish_status = lambda event, **fields: status.append((event, fields))

    verified, measured, left, right, plausible = node._close_for_grasp()

    assert not verified
    assert (measured, left, right, plausible) == transport_lock_sample[1:]
    assert commands == [0.020, 0.0165, 0.010]
    assert status[0][0] == 'gripper_closed'
    assert status[0][1]['close_stage'] == 'transport_lock'
    assert status[0][1]['commanded_position'] == pytest.approx(0.010)
    assert status[0][1]['acquisition_position'] == pytest.approx(0.0155)
    assert status[0][1]['acquisition_plausible_width']
    assert not status[0][1]['transport_lock_engaged']


def test_bilateral_too_narrow_preclose_does_not_command_further_closing():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.gripper_preclose = 0.02
    node.gripper_preload = 0.0165
    node.gripper_transport_lock = 0.010
    node.gripper_closed = 0.0
    node.grasp_min_position = 0.015
    node.grasp_min_margin = 0.001
    node.grasp_max_position = 0.05
    node.joints = {'gripper_left_finger_joint': 0.00002}
    node._left_target_contact_ns = 99
    node._right_target_contact_ns = 88
    commands = []
    status = []
    node._command_gripper = lambda position: commands.append(position) or True
    node._target_contact_sides = lambda: (True, True)
    node._publish_status = lambda event, **fields: status.append((event, fields))

    verified, measured, left, right, plausible = node._close_for_grasp()

    assert not verified
    assert measured == pytest.approx(0.00002)
    assert left and right
    assert not plausible
    # Once both jaws report contact below the minimum retained-book width,
    # moving farther inward cannot produce a valid pinch and risks pushing an
    # already bad contact.  Stop at the guarded pre-close and recover instead.
    assert commands == [0.02]
    assert status == [
        (
            'gripper_closed',
            {
                'close_stage': 'preclose',
                'commanded_position': 0.02,
                'acquisition_position': pytest.approx(0.00002),
                'acquisition_plausible_width': False,
                'transport_lock_engaged': False,
                'measured_position': pytest.approx(0.00002),
                'left_contact': True,
                'right_contact': True,
                'bilateral_contact': True,
                'plausible_width': False,
            },
        )
    ]


def _adaptive_evidence(
    *,
    verified=False,
    reason='bilateral_contact_missing',
    width=0.033,
    left_force=math.nan,
    right_force=math.nan,
    left_samples=0,
    right_samples=0,
):
    return manipulation_node.AdaptiveGraspEvidence(
        verified=verified,
        reason=reason,
        width=width,
        left_force=left_force,
        right_force=right_force,
        effort=0.2,
        effort_delta=0.05,
        left_samples=left_samples,
        right_samples=right_samples,
    )


def _adaptive_close_fixture(start_position, evidence_samples):
    node = object.__new__(manipulation_node.ManipulationNode)
    node.adaptive_close_step = 0.001
    node.adaptive_preload_distance = 0.001
    node.adaptive_confirmation_seconds = 0.12
    node.adaptive_contact_force_minimum = 0.05
    node.adaptive_contact_max_age = 0.15
    node.adaptive_endpoint_tolerance = 0.0001
    node.adaptive_start_tolerance = 0.1
    node.adaptive_unilateral_travel_limit = 0.003
    node.gripper_open = 0.069
    node.gripper_closed = 0.0
    node.gripper_transport_lock = 0.029
    node.grasp_min_position = 0.0285
    node.grasp_min_margin = 0.001
    node.grasp_max_position = 0.05
    node._transport_lock_engaged = False
    node._gripper_open_confirmed = True
    now = SimpleNamespace(nanoseconds=10_000_000_000)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    node._adaptive_effort_baseline = lambda unused: 0.15
    feedback = {'value': manipulation_node.GripperFeedback(
        now.nanoseconds,
        start_position,
        0.0,
        0.2,
    )}
    node._latest_gripper_feedback = lambda: feedback['value']
    commands = []

    def command(position):
        commands.append(position)
        feedback['value'] = manipulation_node.GripperFeedback(
            now.nanoseconds,
            position,
            0.0,
            0.2,
        )
        return True

    node._command_adaptive_gripper_step = command
    evidence = iter(evidence_samples)
    node._adaptive_pressure_evidence = lambda **kwargs: next(evidence)
    clears = []
    node._clear_target_contact_samples = lambda **kwargs: clears.append(
        kwargs
    )
    waits = []
    node._wait_sim_duration = lambda duration: waits.append(duration) or True
    pinch = {'value': (True, start_position, True, True, True)}
    node._pinch_sample = lambda **kwargs: (
        bool(node._transport_lock_engaged and pinch['value'][0]),
        feedback['value'].position,
        pinch['value'][2],
        pinch['value'][3],
        pinch['value'][4],
    )
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )
    return node, commands, clears, waits, pinch, statuses


def test_adaptive_close_stops_at_pressure_then_applies_one_relative_preload():
    model = 'book_col_3_row_2_red'
    missing = _adaptive_evidence(width=0.033)
    acquired = _adaptive_evidence(
        verified=True,
        reason='bilateral_contact_verified',
        width=0.031,
        left_force=0.8,
        right_force=0.9,
        left_samples=3,
        right_samples=3,
    )
    held = _adaptive_evidence(
        verified=True,
        reason='bilateral_contact_verified',
        width=0.030,
        left_force=1.1,
        right_force=1.2,
        left_samples=4,
        right_samples=4,
    )
    samples = (
        (missing, None, None),
        (_adaptive_evidence(width=0.032), None, None),
        (acquired, model, None),
        (held, model, None),
        (held, model, None),
    )
    node, commands, clears, waits, _, statuses = _adaptive_close_fixture(
        0.033,
        samples,
    )

    result = node._adaptive_close_for_grasp()

    assert result == (True, pytest.approx(0.030), True, True, True)
    assert commands == pytest.approx([0.032, 0.031, 0.030])
    assert len(clears) == 2
    assert waits == [pytest.approx(node.adaptive_confirmation_seconds)]
    assert node._transport_lock_engaged
    assert statuses[0][0] == 'gripper_closed'
    assert statuses[0][1]['control_mode'] == 'adaptive_pressure'
    assert statuses[0][1]['reason'] == 'bilateral_pressure_confirmed'
    assert statuses[0][1]['acquisition_position'] == pytest.approx(0.031)
    assert statuses[0][1]['commanded_position'] == pytest.approx(0.030)


def test_adaptive_close_can_seek_from_above_the_plausible_grasp_band():
    model = 'book_col_3_row_2_red'

    def too_wide(width):
        return (
            _adaptive_evidence(reason='width_too_wide', width=width),
            None,
            None,
        )

    acquired = _adaptive_evidence(
        verified=True,
        reason='bilateral_contact_verified',
        width=0.049,
        left_force=0.8,
        right_force=0.9,
        left_samples=3,
        right_samples=3,
    )
    held = _adaptive_evidence(
        verified=True,
        reason='bilateral_contact_verified',
        width=0.048,
        left_force=1.0,
        right_force=1.1,
        left_samples=3,
        right_samples=3,
    )
    samples = (
        too_wide(0.052),
        too_wide(0.051),
        too_wide(0.050),
        (acquired, model, None),
        (held, model, None),
        (held, model, None),
    )
    node, commands, _, _, _, _ = _adaptive_close_fixture(0.052, samples)

    result = node._adaptive_close_for_grasp()

    assert result[0]
    assert commands == pytest.approx([0.051, 0.050, 0.049, 0.048])


def test_adaptive_close_stops_on_pressure_above_the_grasp_band():
    samples = (
        (
            _adaptive_evidence(reason='width_too_wide', width=0.052),
            None,
            None,
        ),
        (
            _adaptive_evidence(
                reason='width_too_wide',
                width=0.051,
                left_force=0.8,
                left_samples=1,
            ),
            'book_col_3_row_2_red',
            'target_not_latched',
        ),
    )
    node, commands, _, _, pinch, statuses = _adaptive_close_fixture(
        0.052,
        samples,
    )
    pinch['value'] = (False, 0.051, True, False, False)

    result = node._adaptive_close_for_grasp()

    assert not result[0]
    assert commands == pytest.approx([0.051])
    assert statuses[0][1]['reason'] == 'contact_above_grasp_width'


@pytest.mark.parametrize(
    ('start_position', 'open_confirmed'),
    ((0.060, True), (0.069, False)),
)
def test_adaptive_close_requires_a_fresh_fully_open_start(
    start_position,
    open_confirmed,
):
    samples = ((
        _adaptive_evidence(reason='width_too_wide', width=start_position),
        None,
        None,
    ),)
    node, commands, _, _, pinch, statuses = _adaptive_close_fixture(
        start_position,
        samples,
    )
    node.adaptive_start_tolerance = 0.0015
    node._gripper_open_confirmed = open_confirmed
    pinch['value'] = (False, start_position, False, False, False)

    result = node._adaptive_close_for_grasp()

    assert not result[0]
    assert commands == []
    assert statuses[0][1]['reason'] == 'open_start_not_confirmed'


def test_adaptive_close_aborts_after_bounded_unilateral_travel():
    def unilateral(width):
        return (
            _adaptive_evidence(
                reason='unilateral_contact',
                width=width,
                left_force=0.8,
                left_samples=3,
            ),
            'book_col_3_row_2_red',
            'target_not_latched',
        )

    samples = [
        (_adaptive_evidence(width=0.035), None, None),
        unilateral(0.034),
        unilateral(0.033),
        unilateral(0.032),
        unilateral(0.031),
    ]
    node, commands, _, _, pinch, statuses = _adaptive_close_fixture(
        0.035,
        samples,
    )
    pinch['value'] = (False, 0.031, True, False, True)

    result = node._adaptive_close_for_grasp()

    assert not result[0]
    assert commands == pytest.approx([0.034, 0.033, 0.032, 0.031])
    assert not node._transport_lock_engaged
    assert statuses[0][1]['reason'] == 'unilateral_contact_travel_limit'


def test_adaptive_close_force_overload_prevents_any_further_closure():
    samples = [(_adaptive_evidence(width=0.033), None, None)]
    node, commands, _, _, pinch, statuses = _adaptive_close_fixture(
        0.033,
        samples,
    )
    pinch['value'] = (False, 0.032, True, False, True)
    command = node._command_adaptive_gripper_step

    def command_with_transient_overload(position):
        commanded = command(position)
        # The evaluator never gets to see this transient sample; the attempt
        # latch must still prevent another inward step.
        node._adaptive_overload_latched = 'force_overload'
        return commanded

    node._command_adaptive_gripper_step = command_with_transient_overload

    result = node._adaptive_close_for_grasp()

    assert not result[0]
    assert commands == pytest.approx([0.032])
    assert statuses[0][1]['reason'] == 'force_overload'


def test_adaptive_close_requires_fresh_pressure_after_preload():
    model = 'book_col_3_row_2_red'
    acquired = _adaptive_evidence(
        verified=True,
        reason='bilateral_contact_verified',
        width=0.031,
        left_force=0.8,
        right_force=0.9,
        left_samples=3,
        right_samples=3,
    )
    command_phase = _adaptive_evidence(
        verified=True,
        reason='bilateral_contact_verified',
        width=0.030,
        left_force=1.0,
        right_force=1.1,
        left_samples=3,
        right_samples=3,
    )
    lost = _adaptive_evidence(
        reason='bilateral_contact_missing',
        width=0.030,
    )
    samples = (
        (_adaptive_evidence(width=0.032), None, None),
        (acquired, model, None),
        (command_phase, model, None),
        (lost, model, None),
    )
    node, commands, clears, waits, pinch, statuses = (
        _adaptive_close_fixture(0.032, samples)
    )
    pinch['value'] = (False, 0.030, False, False, True)

    result = node._adaptive_close_for_grasp()

    assert not result[0]
    assert commands == pytest.approx([0.031, 0.030])
    assert len(clears) == 2
    assert waits == [pytest.approx(node.adaptive_confirmation_seconds)]
    assert not node._transport_lock_engaged
    assert statuses[0][1]['reason'] == 'bilateral_contact_missing'


def test_retained_arm_trajectory_goal_has_ordered_position_only_points():
    node = object.__new__(manipulation_node.ManipulationNode)
    first = np.arange(8, dtype=float)
    second = first + 0.5
    legs = [
        (first, 0.35, 'extension'),
        (second, 0.65, 'compact_transport'),
    ]

    goal, duration = node._make_retained_arm_trajectory_goal(legs)

    assert goal.trajectory.joint_names == list(manipulation_node.ARM_JOINTS)
    assert duration == pytest.approx(1.0)
    assert len(goal.trajectory.points) == 2
    assert goal.trajectory.points[0].positions == pytest.approx(first[1:])
    assert goal.trajectory.points[1].positions == pytest.approx(second[1:])
    assert len(goal.trajectory.points[0].velocities) == 0
    assert len(goal.trajectory.points[1].velocities) == 0
    assert goal.trajectory.points[0].time_from_start.sec == 0
    assert goal.trajectory.points[0].time_from_start.nanosec == 350_000_000
    assert goal.trajectory.points[1].time_from_start.sec == 1
    assert goal.trajectory.points[1].time_from_start.nanosec == 0


def test_retained_arm_trajectory_watchdog_cancels_robot_contact():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._cancel = manipulation_node.threading.Event()
    node._lock = manipulation_node.threading.Lock()
    node._goal_handles = []
    node._target_robot_contact_latched = False
    node.grasp_contact_max_age = 0.75
    node.timeout = 1.0
    node._target_contact_recent = lambda **kwargs: True
    now = SimpleNamespace(nanoseconds=2_000_000_000)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    class DoneFuture:
        def __init__(self, value):
            self.value = value

        def done(self):
            return True

        def result(self):
            return self.value

    class ResultFuture:
        def __init__(self):
            self.calls = 0

        def done(self):
            self.calls += 1
            if self.calls == 1:
                node._target_robot_contact_latched = True
                return False
            return True

        def result(self):
            return SimpleNamespace(status=manipulation_node.GoalStatus.STATUS_CANCELED)

    result_future = ResultFuture()
    goal_handle = SimpleNamespace(accepted=True, cancellations=0)
    goal_handle.get_result_async = lambda: result_future

    def cancel():
        goal_handle.cancellations += 1
        return DoneFuture(SimpleNamespace())

    goal_handle.cancel_goal_async = cancel
    node.arm_client = SimpleNamespace(
        send_goal_async=lambda goal: DoneFuture(goal_handle)
    )
    legs = [(np.zeros(8), 0.35, 'supported_cradle_extension')]

    succeeded, contact_lost = node._send_retained_arm_trajectory(
        object(),
        0.35,
        legs,
        'compact_transport',
    )

    assert not succeeded
    assert contact_lost
    assert goal_handle.cancellations == 1
    assert node._goal_handles == []
    assert statuses == [
        (
            'grasp_lost',
            {
                'command': 'compact_transport',
                'phase': 'supported_cradle_extension',
                'leg': 0,
                'reason': 'payload_robot_contact',
            },
        )
    ]


def test_retained_arm_trajectory_watchdog_cancels_contact_loss():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._cancel = manipulation_node.threading.Event()
    node._lock = manipulation_node.threading.Lock()
    node._goal_handles = []
    node._target_robot_contact_latched = False
    node.grasp_contact_max_age = 0.75
    node.timeout = 1.0
    contacts = iter((True, False))
    node._target_contact_recent = lambda **kwargs: next(contacts)
    now = SimpleNamespace(nanoseconds=2_000_000_000)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    class DoneFuture:
        def __init__(self, value):
            self.value = value

        def done(self):
            return True

        def result(self):
            return self.value

    class ResultFuture:
        def __init__(self):
            self.calls = 0

        def done(self):
            self.calls += 1
            return self.calls >= 2

        def result(self):
            return SimpleNamespace(status=manipulation_node.GoalStatus.STATUS_CANCELED)

    result_future = ResultFuture()
    goal_handle = SimpleNamespace(accepted=True, cancellations=0)
    goal_handle.get_result_async = lambda: result_future

    def cancel():
        goal_handle.cancellations += 1
        return DoneFuture(SimpleNamespace())

    goal_handle.cancel_goal_async = cancel
    node.arm_client = SimpleNamespace(
        send_goal_async=lambda goal: DoneFuture(goal_handle)
    )
    legs = [(np.zeros(8), 0.35, 'supported_cradle_lowering')]

    succeeded, contact_lost = node._send_retained_arm_trajectory(
        object(),
        0.35,
        legs,
        'compact_transport',
    )

    assert not succeeded
    assert contact_lost
    assert goal_handle.cancellations == 1
    assert node._goal_handles == []
    assert statuses[0][1]['reason'] == 'contact_lost'


def test_retained_endpoint_wait_allows_delayed_joint_state_convergence(monkeypatch):
    node = object.__new__(manipulation_node.ManipulationNode)
    node._cancel = threading.Event()
    node.grasp_contact_max_age = 0.75
    node.timeout = 1.0
    target = np.zeros(8)
    intermediate = target.copy()
    intermediate[1] = 0.02
    samples = iter((intermediate, target.copy()))
    node._measured_left_solution = lambda: next(samples)
    hazards = []
    node._payload_hazard_reason = (
        lambda **kwargs: hazards.append(kwargs['max_age']) or None
    )
    clock_ns = [0]
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=clock_ns[0])
    )
    monkeypatch.setattr(
        manipulation_node.time,
        'sleep',
        lambda duration: clock_ns.__setitem__(
            0, clock_ns[0] + int(duration * 1e9)
        ),
    )
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    measured = node._wait_for_retained_endpoint(
        target,
        command='compact_transport',
        phase='post_retreat_clearance_extension',
        leg=1,
        settle_timeout=0.5,
    )

    np.testing.assert_allclose(measured, target)
    assert clock_ns[0] == 20_000_000
    assert hazards == pytest.approx([0.15, 0.15, 0.15])
    assert statuses == []


def test_retained_endpoint_wait_fails_closed_on_payload_hazard(monkeypatch):
    node = object.__new__(manipulation_node.ManipulationNode)
    node._cancel = threading.Event()
    node.grasp_contact_max_age = 0.75
    node.timeout = 1.0
    target = np.zeros(8)
    intermediate = target.copy()
    intermediate[1] = 0.02
    measurements = []
    node._measured_left_solution = (
        lambda: measurements.append(True) or intermediate.copy()
    )
    hazards = iter((None, 'payload_robot_contact'))
    node._payload_hazard_reason = lambda **kwargs: next(hazards)
    clock_ns = [0]
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=clock_ns[0])
    )
    monkeypatch.setattr(
        manipulation_node.time,
        'sleep',
        lambda duration: clock_ns.__setitem__(
            0, clock_ns[0] + int(duration * 1e9)
        ),
    )
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    assert node._wait_for_retained_endpoint(
        target,
        command='compact_transport',
        phase='post_retreat_cradle_roll',
        leg=3,
        settle_timeout=0.5,
    ) is None

    assert measurements == [True]
    assert statuses == [
        (
            'grasp_lost',
            {
                'command': 'compact_transport',
                'phase': 'post_retreat_cradle_roll',
                'reason': 'payload_robot_contact',
                'endpoint_settle': True,
                'leg': 3,
            },
        )
    ]


def test_retained_endpoint_wait_stops_at_simulated_timeout(monkeypatch):
    node = object.__new__(manipulation_node.ManipulationNode)
    node._cancel = threading.Event()
    node.grasp_contact_max_age = 0.75
    node.timeout = 1.0
    target = np.zeros(8)
    intermediate = target.copy()
    intermediate[1] = 0.02
    measurements = []
    node._measured_left_solution = (
        lambda: measurements.append(True) or intermediate.copy()
    )
    node._payload_hazard_reason = lambda **kwargs: None
    clock_ns = [0]
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=clock_ns[0])
    )
    monkeypatch.setattr(
        manipulation_node.time,
        'sleep',
        lambda duration: clock_ns.__setitem__(
            0, clock_ns[0] + 200_000_000
        ),
    )
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    assert node._wait_for_retained_endpoint(
        target,
        command='compact_transport',
        phase='compact_transport_final',
        settle_timeout=0.3,
    ) is None

    assert len(measurements) == 3
    assert statuses[-1][0] == 'trajectory_endpoint_missed'
    assert statuses[-1][1]['reason'] == 'simulated_settle_timeout'
    assert statuses[-1][1]['arm_error'] == pytest.approx(0.02)


def test_unresolved_controller_goal_interlocks_the_next_command():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._lock = manipulation_node.threading.Lock()
    node._busy = False
    node._goal_handles = [object()]
    node._cancel = manipulation_node.threading.Event()
    node._cancel.set()
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    node._on_command(
        SimpleNamespace(data=manipulation_node.encode_event('stow'))
    )

    assert node._cancel.is_set()
    assert not node._busy
    assert statuses == [
        (
            'rejected',
            {'command': 'stow', 'reason': 'controller_goal_unresolved'},
        )
    ]


def test_follow_keeps_unconfirmed_cancelled_goal_interlocked(monkeypatch):
    node = object.__new__(manipulation_node.ManipulationNode)
    node._lock = manipulation_node.threading.Lock()
    node._cancel = manipulation_node.threading.Event()
    node._goal_handles = []
    node.timeout = 1.0

    class DoneFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class NeverDoneFuture:
        def done(self):
            return False

    result_future = NeverDoneFuture()
    goal_handle = SimpleNamespace(accepted=True, cancellations=0)
    goal_handle.get_result_async = lambda: result_future

    def cancel():
        goal_handle.cancellations += 1
        return DoneFuture(SimpleNamespace())

    goal_handle.cancel_goal_async = cancel
    send_future = DoneFuture(goal_handle)
    client = SimpleNamespace(
        wait_for_server=lambda **kwargs: True,
        send_goal_async=lambda goal: send_future,
    )

    def wait_future(future, timeout):
        if future is result_future:
            raise TimeoutError('simulated trajectory timeout')
        return future.result()

    node._wait_future = wait_future
    times = iter((0.0, 10.0, 10.0, 20.0))
    monkeypatch.setattr(
        manipulation_node.time,
        'monotonic',
        lambda: next(times),
    )

    with pytest.raises(
        RuntimeError,
        match='trajectory cancellation was not confirmed',
    ):
        node._follow(client, ['joint'], [0.0], 0.5)

    assert goal_handle.cancellations == 1
    assert node._goal_handles == [goal_handle]
    assert node._cancel.is_set()


def test_follow_rejects_fast_cradle_when_payload_collision_is_already_latched():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._lock = manipulation_node.threading.Lock()
    node._cancel = manipulation_node.threading.Event()
    node._goal_handles = []
    node.timeout = 1.0
    node._payload_robot_watchdog_enabled = True
    node._target_robot_contact_latched = True
    node._payload_hazard_latched = None
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    calls = []
    client = SimpleNamespace(
        wait_for_server=lambda **kwargs: calls.append('wait') or True,
        send_goal_async=lambda goal: calls.append('send'),
    )

    assert not node._follow(client, ['arm_left_7_joint'], [-1.10], 0.75)
    assert calls == []
    assert node._goal_handles == []
    assert node._payload_hazard_latched == 'payload_robot_contact'
    assert statuses == [
        ('payload_hazard', {'reason': 'payload_robot_contact'}),
    ]


def test_follow_cancels_fast_cradle_when_payload_collision_latches_in_flight():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._lock = manipulation_node.threading.Lock()
    node._cancel = manipulation_node.threading.Event()
    node._goal_handles = []
    node.timeout = 1.0
    node._payload_robot_watchdog_enabled = True
    node._target_robot_contact_latched = False
    node._payload_hazard_latched = None
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    class ImmediateFuture:
        def __init__(self, value):
            self.value = value

        def done(self):
            return True

        def result(self):
            return self.value

    class ResultFuture:
        def __init__(self):
            self.terminal = False

        def done(self):
            if not self.terminal:
                node._target_robot_contact_latched = True
            return self.terminal

        def result(self):
            return SimpleNamespace(
                status=manipulation_node.GoalStatus.STATUS_CANCELED
            )

    result_future = ResultFuture()
    goal_handle = SimpleNamespace(accepted=True, cancellations=0)
    goal_handle.get_result_async = lambda: result_future

    def cancel():
        goal_handle.cancellations += 1
        result_future.terminal = True
        return ImmediateFuture(SimpleNamespace())

    goal_handle.cancel_goal_async = cancel
    client = SimpleNamespace(
        wait_for_server=lambda **kwargs: True,
        send_goal_async=lambda goal: ImmediateFuture(goal_handle),
    )

    assert not node._follow(client, ['arm_left_7_joint'], [-1.10], 0.75)
    assert goal_handle.cancellations == 1
    assert node._goal_handles == []
    assert node._payload_hazard_latched == 'payload_robot_contact'
    assert statuses == [
        ('payload_hazard', {'reason': 'payload_robot_contact'}),
    ]


def test_follow_does_not_let_terminal_success_beat_payload_collision_latch():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._lock = manipulation_node.threading.Lock()
    node._cancel = manipulation_node.threading.Event()
    node._goal_handles = []
    node.timeout = 1.0
    node._payload_robot_watchdog_enabled = True
    node._target_robot_contact_latched = False
    node._payload_hazard_latched = None
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    class ImmediateFuture:
        def __init__(self, value):
            self.value = value

        def done(self):
            return True

        def result(self):
            return self.value

    class SimultaneousResultFuture:
        def done(self):
            node._target_robot_contact_latched = True
            return True

        def result(self):
            return SimpleNamespace(
                status=manipulation_node.GoalStatus.STATUS_SUCCEEDED
            )

    result_future = SimultaneousResultFuture()
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: result_future,
    )
    client = SimpleNamespace(
        wait_for_server=lambda **kwargs: True,
        send_goal_async=lambda goal: ImmediateFuture(goal_handle),
    )

    assert not node._follow(client, ['arm_left_7_joint'], [-1.10], 0.75)
    assert node._goal_handles == []
    assert node._payload_hazard_latched == 'payload_robot_contact'
    assert statuses == [
        ('payload_hazard', {'reason': 'payload_robot_contact'}),
    ]


def _stub_retained_leg_dispatch(node):
    """Keep sentinel-pose ordering fixtures separate from action safety tests."""
    node._make_retained_arm_trajectory_goal = lambda legs: (
        tuple(legs), sum(duration for _, duration, _ in legs)
    )

    def send(goal, duration, legs, command, *, leg_offset=0, initial_pressure_gate=None):
        assert len(legs) == 1
        assert len(goal) == 1
        assert duration == legs[0][1]
        solution, leg_duration, _ = legs[0]
        # Resolve the current stub at call time: some fixtures wrap it later
        # to verify when the loaded move occurs relative to approach/recovery.
        move = lambda: bool(node._move_arm_solution(solution, leg_duration))
        result = initial_pressure_gate.send(move) if initial_pressure_gate is not None else move()
        return result, False

    node._send_retained_arm_trajectory = send


def test_retention_gate_stops_on_the_first_failed_carried_leg():
    node = object.__new__(manipulation_node.ManipulationNode)
    _stub_retained_leg_dispatch(node)
    moved = []
    status = []
    contacts = iter((True, True, False))
    node._move_arm_solution = lambda solution, duration: moved.append(
        (np.asarray(solution), duration)
    ) or True
    node._target_contact_recent = lambda **kwargs: next(contacts)
    node._publish_status = lambda event, **fields: status.append((event, fields))
    legs = [
        (np.zeros(8), 0.65, 'extraction'),
        (np.ones(8), 0.65, 'fixed_orientation_lowering'),
        (np.full(8, 2.0), 2.8, 'home'),
        (np.full(8, 3.0), 2.8, 'must_not_run'),
    ]

    succeeded, next_leg, contact_lost = node._execute_retained_arm_legs(
        legs,
        'pick',
    )

    assert not succeeded
    assert contact_lost
    assert next_leg == 3
    assert len(moved) == 3
    assert status == [
        (
            'grasp_lost',
            {
                'command': 'pick',
                'phase': 'home',
                'leg': 2,
                'reason': 'contact_lost',
            },
        )
    ]


def test_retained_leg_exception_is_reported_as_recoverable_motion_failure():
    node = object.__new__(manipulation_node.ManipulationNode)
    _stub_retained_leg_dispatch(node)
    status = []
    node._move_arm_solution = lambda solution, duration: (_ for _ in ()).throw(
        TimeoutError('controller timed out')
    )
    node._publish_status = lambda event, **fields: status.append((event, fields))

    succeeded, next_leg, contact_lost = node._execute_retained_arm_legs(
        [(np.zeros(8), 0.65, 'extraction')],
        'pick',
    )

    assert not succeeded
    assert next_leg == 0
    assert not contact_lost
    assert status[0][0] == 'motion_exception'
    assert status[0][1]['reason'] == 'controller timed out'


def test_retention_after_leg_rejects_contact_stale_at_motion_endpoint():
    node = object.__new__(manipulation_node.ManipulationNode)
    node.grasp_contact_max_age = 0.75
    node._gravity_supported_payload = False
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)
    )
    node._left_target_contact_ns = 800_000_000
    node._right_target_contact_ns = 800_000_000
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    assert not node._retention_after_leg('pick', 'extraction', 2)
    assert statuses[-1] == (
        'grasp_lost',
        {
            'command': 'pick',
            'phase': 'extraction',
            'leg': 2,
            'reason': 'contact_lost',
        },
    )

    node._left_target_contact_ns = 900_000_000
    node._right_target_contact_ns = 900_000_000
    assert node._retention_after_leg('pick', 'extraction', 3)


def test_ambiguous_close_never_runs_unloaded_recovery_if_reopen_fails():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = None
    node._open_gripper = lambda: False
    unloaded_calls = []
    node._recover_unloaded_pick = lambda *args: unloaded_calls.append(True) or True
    attached = np.zeros((8, 3))

    with pytest.raises(RuntimeError, match='pick_recovery_failed'):
        node._recover_ambiguous_grasp(attached, [np.zeros(8)], [])

    assert unloaded_calls == []
    assert np.array_equal(node._held_book_corners, attached)


def test_ambiguous_close_never_opens_after_known_payload_robot_contact():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._lock = threading.Lock()
    node._target_robot_contact_latched = True
    node._held_book_corners = None
    opened = []
    moved = []
    statuses = []
    node._open_gripper = lambda: opened.append(True) or True
    node._move_arm_solution = lambda *args: moved.append('arm') or True
    node._move_torso = lambda *args, **kwargs: moved.append('torso') or True
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )
    attached = np.zeros((8, 3))

    with pytest.raises(RuntimeError, match='pick_payload_lost'):
        node._recover_ambiguous_grasp(attached, [np.zeros(8)], [])

    assert opened == []
    assert moved == []
    assert np.array_equal(node._held_book_corners, attached)
    assert statuses == [
        (
            'carried_recovery',
            {
                'command': 'pick',
                'cause': 'payload_robot_contact',
                'retained_before_recovery': False,
                'gripper_opened': False,
                'recovery_succeeded': False,
                'recovery_halted': True,
                'retained_stop': True,
            },
        )
    ]


def test_place_motion_failure_stops_with_retained_payload_instead_of_dropping():
    node = object.__new__(manipulation_node.ManipulationNode)
    attached = np.zeros((8, 3))
    node._held_book_corners = attached.copy()
    opened = []
    node._open_gripper = lambda: opened.append(True) or True
    status = []
    node._publish_status = lambda event, **fields: status.append((event, fields))

    with pytest.raises(RuntimeError, match='place_recovery_failed'):
        node._recover_closed_place(
            cause='motion_failed',
            remaining_approach_legs=(),
            cartesian_solutions=(),
            carried_transition_waypoints=(),
            carried_start=np.zeros(8),
            unloaded_home_waypoints=(),
            release_allowed=False,
        )

    assert opened == []
    assert np.array_equal(node._held_book_corners, attached)
    assert status[-1][1]['retained_stop']


@pytest.mark.parametrize('cause', ('contact_lost', 'contact_lost_after_torso'))
def test_place_contact_loss_halts_closed_without_arm_motion(cause):
    node = object.__new__(manipulation_node.ManipulationNode)
    trace = []
    node._open_gripper = lambda: trace.append(('open',)) or True
    node._move_arm_solution = lambda *args: trace.append(('arm',)) or True
    node._move_torso = lambda *args, **kwargs: trace.append(('torso',)) or True
    status = []
    node._publish_status = lambda event, **fields: status.append((event, fields))

    with pytest.raises(RuntimeError, match='place_payload_lost'):
        node._recover_closed_place(
            cause=cause,
            remaining_approach_legs=((np.ones(8), 0.5, 'bin_transition'),),
            cartesian_solutions=(np.ones(8),),
            carried_transition_waypoints=(np.ones(8),),
            carried_start=np.zeros(8),
            unloaded_home_waypoints=(np.ones(8),),
        )

    assert trace == []
    assert status == [
        (
            'carried_recovery',
            {
                'command': 'place',
                'cause': cause,
                'gripper_opened': False,
                'recovery_succeeded': False,
                'recovery_halted': True,
                'retained_stop': True,
            },
        )
    ]


def _sentinel_pick_accuracy(positions, rotation, solutions):
    """Separate command-order fixtures from real-URDF accuracy regressions."""
    return {
        'actual_grasp_position': np.asarray(positions[-1]).tolist(),
        'actual_grasp_orientation': np.asarray(rotation).tolist(),
        'grasp_position_error_base_m': [0., 0., 0.],
        'grasp_position_error_m': 0.,
        'grasp_orientation_error_rad': 0.,
        'approach_max_position_error_m': 0.,
        'approach_max_orientation_error_rad': 0.,
    }


def test_retained_route_motion_failure_halts_at_unknown_controller_pose(monkeypatch):
    # This recovery fixture uses sentinel joint vectors, not geometric poses.
    monkeypatch.setattr(manipulation_node, 'check_open_gripper_approach',
                        lambda *args, **kwargs: SimpleNamespace(ok=True))
    node = object.__new__(manipulation_node.ManipulationNode)
    _stub_retained_leg_dispatch(node)
    node._pick_path_accuracy = _sentinel_pick_accuracy
    node.gripper_open = 0.069
    node.grasp_depth_offset = 0.08
    node.top_row_grasp_depth_offset = 0.060
    node.top_row_grasp_vertical_offset = -0.015
    node.top_row_grasp_lateral_offset = -0.001
    node.top_row_loaded_clearance_lift = 0.018
    node.pregrasp_offset = 0.14
    node.cartesian_clearance = 0.45
    node.top_row_cartesian_clearance = 0.46
    node.cartesian_step = 0.06
    node.pick_torso_height = 0.35

    front = np.asarray([0.70, -0.05, 1.25])
    cartesian = [
        np.full(8, 10.0, dtype=float),
        np.full(8, 11.0, dtype=float),
    ]
    lowering = [np.full(8, 20.0, dtype=float)]
    home_route = [np.full(8, 30.0, dtype=float)]
    node._wait_for_perception_point = lambda attribute: front.copy()
    node._solve_cartesian_path = lambda *args, **kwargs: (
        cartesian,
        0,
        1.0,
        [],
    )
    cached_plan = {'sentinel': 'preflighted-before-open'}

    def plan_carried_return(*args, **kwargs):
        node._cached_post_retreat_plan = cached_plan
        return (
            lowering,
            [],
            None,
            home_route,
            np.zeros((8, 3)),
        )

    node._plan_carried_return = plan_carried_return
    node._carried_robot_transition_is_safe = lambda *args, **kwargs: True
    node._publish_status = lambda *args, **kwargs: None

    gripper_commands = []
    held = {'value': False}

    def command_gripper(position, *, respect_cancel=False):
        assert respect_cancel if position == node.gripper_open else True
        gripper_commands.append(position)
        if position == pytest.approx(node.gripper_open):
            held['value'] = False
        return True

    def close_for_grasp():
        held['value'] = True
        return True, 0.025, True, True, True

    node._command_gripper = command_gripper
    node._close_for_grasp = close_for_grasp
    torso_moves = []
    node._move_torso = lambda height, duration, **kwargs: torso_moves.append(height) or True

    arm_moves = []

    def move_arm(solution, duration):
        # The real state-clearing pre-grasp open has run by the first approach
        # move, but the route is installed only after acquisition succeeds.
        if not held['value']:
            assert node._cached_post_retreat_plan is None
        else:
            assert node._cached_post_retreat_plan is cached_plan
        value = int(np.asarray(solution, dtype=float)[1])
        arm_moves.append(value)
        # The first two moves are the approach.  Fail the first carried
        # extraction command once, as an action abort can leave the book held
        # at an unknown point along that already checked segment.
        return len(arm_moves) != 3

    node._move_arm_solution = move_arm
    node._target_contact_recent = lambda **kwargs: True
    node._fresh_retention_probe = lambda *args, **kwargs: True

    guard = _WorkflowEmptyPickupGuard(node)
    monkeypatch.setattr(empty_pickup.EmptyPickupCollision, 'capture', lambda actual: guard)
    with pytest.raises(RuntimeError, match='pick_recovery_failed'):
        node._pick()

    # The controller may have stopped anywhere inside the failed segment.
    # A fresh pinch keeps the gripper closed, but no unmeasured recovery arm
    # or torso motion is allowed.
    assert gripper_commands == [node.gripper_open]
    assert held['value']
    assert arm_moves == [10, 11, 10]
    assert torso_moves == [pytest.approx(node.pick_torso_height)]


def test_pick_motion_failure_opens_and_halts_when_fresh_probe_fails():
    node = object.__new__(manipulation_node.ManipulationNode)
    trace = []
    node._fresh_retention_probe = lambda *args, **kwargs: False
    node._move_arm_solution = lambda solution, duration: trace.append(
        ('arm', int(np.asarray(solution)[0]))
    ) or True
    node._move_torso = lambda height, duration, **kwargs: trace.append(
        ('torso', height)
    ) or True
    node._open_gripper = lambda: trace.append(('open',)) or True
    status = []
    node._publish_status = lambda event, **fields: status.append((event, fields))

    with pytest.raises(RuntimeError, match='pick_payload_lost'):
        node._recover_closed_pick(
            cause='motion_failed',
            remaining_carried_legs=(
                (np.full(8, 1.0), 0.5, 'first'),
                (np.full(8, 2.0), 0.5, 'second'),
            ),
            unloaded_recovery_route=(np.full(8, 3.0),),
        )

    assert trace == [
        ('open',),
    ]
    assert status == [(
        'carried_recovery',
        {
            'command': 'pick',
            'cause': 'motion_failed',
            'retained_before_recovery': False,
            'gripper_opened': True,
            'recovery_succeeded': False,
            'recovery_halted': True,
        },
    )]


def test_pick_recovery_rechecks_robot_collision_after_fresh_probe():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._lock = threading.Lock()
    node._target_robot_contact_latched = False
    node._gravity_supported_payload = False
    actions = []

    def probe(*args, **kwargs):
        node._target_robot_contact_latched = True
        return False

    node._fresh_retention_probe = probe
    node._open_gripper = lambda: actions.append('open') or True
    node._move_arm_solution = lambda *args: actions.append('arm') or True
    node._move_torso = lambda *args, **kwargs: actions.append('torso') or True
    statuses = []
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    with pytest.raises(RuntimeError, match='pick_payload_lost'):
        node._recover_closed_pick(
            cause='motion_failed',
            remaining_carried_legs=(
                (np.ones(8), 0.5, 'must_not_run'),
            ),
            unloaded_recovery_route=(np.ones(8),),
        )

    assert actions == []
    assert statuses == [
        (
            'carried_recovery',
            {
                'command': 'pick',
                'cause': 'payload_robot_contact',
                'retained_before_recovery': False,
                'gripper_opened': False,
                'recovery_succeeded': False,
                'recovery_halted': True,
                'retained_stop': True,
            },
        )
    ]


@pytest.mark.parametrize(
    ('cause', 'recent_contact'),
    (
        ('final_grasp_settle_failed', True),
        ('final_grasp_settle_failed', False),
        ('final_grasp_width_invalid', True),
    ),
)
def test_ambiguous_final_grasp_gate_stops_closed(cause, recent_contact):
    node = object.__new__(manipulation_node.ManipulationNode)
    node._target_robot_contact_latched = False
    node._target_contact_recent = lambda **kwargs: recent_contact
    actions = []
    statuses = []
    node._open_gripper = lambda: actions.append('open') or True
    node._move_arm_solution = lambda *args: actions.append('arm') or True
    node._move_torso = lambda *args, **kwargs: actions.append('torso') or True
    node._publish_status = lambda event, **fields: statuses.append(
        (event, fields)
    )

    with pytest.raises(RuntimeError, match='pick_recovery_failed'):
        node._recover_closed_pick(
            cause=cause,
            remaining_carried_legs=((np.ones(8), 0.5, 'must_not_run'),),
            unloaded_recovery_route=(np.ones(8),),
        )

    assert actions == []
    assert statuses == [
        (
            'carried_recovery',
            {
                'command': 'pick',
                'cause': cause,
                'retained_before_recovery': recent_contact,
                'gripper_opened': False,
                'recovery_succeeded': False,
                'recovery_halted': True,
                'retained_stop': True,
            },
        )
    ]


@pytest.mark.parametrize(
    'cause',
    ('contact_lost', 'cradle_contact_verification_failed'),
)
def test_pick_recovery_does_not_move_after_failed_initial_reopen(cause):
    node = object.__new__(manipulation_node.ManipulationNode)
    # A callback can arrive between the failed gate and recovery.  Confirmed
    # loss causes must still fail closed instead of re-sampling it as retained.
    node._target_contact_recent = lambda **kwargs: True
    node._open_gripper = lambda: False
    moves = []
    node._move_arm_solution = lambda *args: moves.append('arm') or True
    node._move_torso = lambda *args, **kwargs: moves.append('torso') or True
    node._publish_status = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match='pick_payload_lost'):
        node._recover_closed_pick(
            cause=cause,
            remaining_carried_legs=((np.zeros(8), 0.5, 'extraction'),),
            unloaded_recovery_route=(np.ones(8),),
        )

    assert moves == []


@pytest.mark.parametrize(
    'cause',
    ('cradle_contact_verification_failed', 'motion_failed'),
)
def test_supported_pick_recovery_halts_without_opening_over_the_arm(cause):
    node = object.__new__(manipulation_node.ManipulationNode)
    node._gravity_supported_payload = True
    opened = []
    moved = []
    status = []
    node._open_gripper = lambda: opened.append(True) or True
    node._move_arm_solution = lambda *args: moved.append('arm') or True
    node._move_torso = lambda *args, **kwargs: moved.append('torso') or True
    node._fresh_retention_probe = lambda *args, **kwargs: False
    node._publish_status = lambda event, **fields: status.append((event, fields))

    with pytest.raises(RuntimeError, match='pick_payload_lost'):
        node._recover_closed_pick(
            cause=cause,
            remaining_carried_legs=(),
            unloaded_recovery_route=(),
        )

    assert opened == []
    assert moved == []
    assert status == [
        (
            'carried_recovery',
            {
                'command': 'pick',
                'cause': cause,
                'retained_before_recovery': False,
                'gripper_opened': False,
                'recovery_succeeded': False,
                'recovery_halted': True,
                'retained_stop': True,
            },
        )
    ]


def _mocked_pick_trace(
    front,
    *,
    grasp_verified=True,
    settle_succeeded=True,
    final_contact=True,
    cradle_failure=None,
    configure_node=None,
    approach_guard=None,
    empty_guard_failure=None,
):
    node = object.__new__(manipulation_node.ManipulationNode)
    _stub_retained_leg_dispatch(node)
    node._pick_path_accuracy = _sentinel_pick_accuracy
    node.gripper_open = 0.069
    node.carried_shelf_retreat_clearance = 0.25
    node.grasp_depth_offset = 0.08
    node.top_row_grasp_depth_offset = 0.060
    node.top_row_grasp_vertical_offset = -0.015
    node.top_row_grasp_lateral_offset = -0.001
    node.top_row_loaded_clearance_lift = 0.018
    node.pregrasp_offset = 0.14
    node.cartesian_clearance = 0.45
    node.top_row_cartesian_clearance = 0.46
    node.cartesian_step = 0.06
    node.pick_torso_height = 0.35
    node.grasp_min_position = 0.015
    node.grasp_min_margin = 0.001
    node.grasp_max_position = 0.05
    node.gripper_settle = 1.0
    node.grasp_contact_max_age = 0.75
    node.carried_cradle_transfer = 0.75
    node.joints = {'gripper_left_finger_joint': 0.018}
    node.arm_client = object()

    front = np.asarray(front, dtype=float)
    cartesian = [
        np.full(8, 10.0, dtype=float),
        np.full(8, 11.0, dtype=float),
        np.full(8, 12.0, dtype=float),
    ]
    lowering = [np.full(8, 20.0, dtype=float)]
    home_route = [np.full(8, 30.0, dtype=float)]
    node._wait_for_perception_point = lambda attribute: front.copy()
    solve = {}

    def solve_cartesian_path(*args, **kwargs):
        solve['endpoint_first'] = kwargs.get('endpoint_first', False)
        solve['position_tolerance'] = kwargs.get('position_tolerance')
        solve['orientation_tolerance'] = kwargs.get('orientation_tolerance')
        solve['positions'] = [np.asarray(value, dtype=float) for value in args[0]]
        # Mirror the real solver's setup callback followed by full-candidate
        # admission. Keep sentinel IK separate from actual geometry tests.
        planner = kwargs.get('setup_transition_planner')
        if planner is not None:
            assert callable(kwargs.get('transition_edge_validator'))
            assert callable(kwargs.get('candidate_validator'))
            transition = planner(kwargs['transition_start'], cartesian[0])
            if transition is None or not kwargs['candidate_validator'](cartesian, transition):
                raise RuntimeError('fixture_empty_candidate_rejected')
        return cartesian, 0, 1.0, []

    node._solve_cartesian_path = solve_cartesian_path
    plan = {}

    def plan_carried_return(
        observed_front,
        grasp_solution,
        clearance_solution,
        rotation,
        torso_height,
    ):
        plan['clearance'] = np.asarray(clearance_solution, dtype=float).copy()
        if front[2] >= 1.42:
            node._cached_post_retreat_plan = {
                'requires_gravity_support': False,
                'sentinel': 'supported_post_retreat_return',
            }
            return [], [], None, [], np.zeros((8, 3))
        return lowering, [], None, home_route, np.zeros((8, 3))

    node._plan_carried_return = plan_carried_return
    node._carried_robot_transition_is_safe = lambda *args, **kwargs: True
    status = []
    node._publish_status = lambda event, **fields: status.append((event, fields))
    gripper_commands = []
    gripper_admission = []
    def command_gripper(position, *, respect_cancel=False):
        gripper_admission.append((position, respect_cancel))
        gripper_commands.append(position)
        return True
    node._command_gripper = command_gripper
    node._close_for_grasp = lambda: (
        grasp_verified,
        0.018 if grasp_verified else 0.0,
        grasp_verified,
        grasp_verified,
        grasp_verified,
    )
    torso_moves = []
    node._move_torso = lambda height, duration, **kwargs: torso_moves.append(height) or True
    arm_moves = []
    arm_durations = []
    roll_guard_states = []

    def move_arm(solution, duration):
        value = int(np.asarray(solution, dtype=float)[1])
        arm_moves.append(value)
        arm_durations.append(float(duration))
        if value == 21:
            roll_guard_states.append(
                (
                    bool(getattr(node, '_retention_probe_active', False)),
                    bool(getattr(node, '_gravity_supported_payload', False)),
                    bool(
                        getattr(
                            node,
                            '_payload_robot_watchdog_enabled',
                            False,
                        )
                    ),
                )
            )
            if cradle_failure == 'raise':
                raise RuntimeError('simulated cradle controller exception')
            if cradle_failure == 'return_false':
                return False
        return True

    node._move_arm_solution = move_arm
    home_follows = []
    node._follow = lambda client, names, positions, duration: home_follows.append(
        tuple(positions)
    ) or True
    node._target_contact_recent = lambda **kwargs: True
    node._target_contact_sides = lambda **kwargs: (True, True)
    fresh_probes = []
    node._fresh_retention_probe = lambda command, phase, **kwargs: (
        fresh_probes.append((command, phase, kwargs)) or True
    )
    settle_calls = []
    final_probe_states = []

    def wait_sim_duration(duration):
        final_probe_states.append(
            ('wait', bool(getattr(node, '_retention_probe_active', False)))
        )
        settle_calls.append(duration)
        return settle_succeeded

    node._wait_sim_duration = wait_sim_duration
    pinch_contacts = iter([final_contact])

    def pinch_sample(**kwargs):
        final_probe_states.append(
            ('sample', bool(getattr(node, '_retention_probe_active', False)))
        )
        contact = next(pinch_contacts)
        return (
            contact,
            0.018,
            contact,
            contact,
            True,
        )

    node._pinch_sample = pinch_sample
    node._plan_retracted_transition = lambda *args, **kwargs: []
    retained_phases = []
    node._retention_after_leg = (
        lambda command, phase, leg: retained_phases.append(phase) or True
    )
    closed_recoveries = []

    def recover_closed_pick(**fields):
        closed_recoveries.append(
            {
                **fields,
                'gravity_supported_at_recovery': bool(
                    getattr(node, '_gravity_supported_payload', False)
                ),
                'watchdog_enabled_at_recovery': bool(
                    getattr(node, '_payload_robot_watchdog_enabled', False)
                ),
                'probe_active_at_recovery': bool(
                    getattr(node, '_retention_probe_active', False)
                ),
            }
        )
        node._held_book_corners = None
        return False

    node._recover_closed_pick = recover_closed_pick

    if configure_node is not None:
        configure_node(node)
    # Sentinel joint vectors test command ordering independently of FK. The
    # real mesh guard has separate official-URDF tests; callers can inject an
    # explicit rejection here to verify the production integration.
    empty_guard = _WorkflowEmptyPickupGuard(node, empty_guard_failure)
    def capture_empty_guard(actual_node):
        assert actual_node is node
        empty_guard.calls.append(('capture',))
        return empty_guard
    with patch.object(manipulation_node, 'check_open_gripper_approach',
                      approach_guard or (lambda *args, **kwargs: SimpleNamespace(ok=True))), \
            patch.object(empty_pickup.EmptyPickupCollision, 'capture', capture_empty_guard):
        succeeded = node._pick()
    return {
        'succeeded': succeeded,
        'solve': solve,
        'plan': plan,
        'arm_moves': arm_moves,
        'arm_durations': arm_durations,
        'roll_guard_states': roll_guard_states,
        'torso_moves': torso_moves,
        'gripper_commands': gripper_commands,
        'gripper_admission': gripper_admission,
        'empty_geometry_calls': empty_guard.calls,
        'home_follows': home_follows,
        'status': status,
        'retained_phases': retained_phases,
        'fresh_probes': fresh_probes,
        'settle_calls': settle_calls,
        'final_probe_states': final_probe_states,
        'retention_probe_active': getattr(
            node,
            '_retention_probe_active',
            False,
        ),
        'payload_robot_watchdog_enabled': getattr(
            node,
            '_payload_robot_watchdog_enabled',
            False,
        ),
        'closed_recoveries': closed_recoveries,
        'carried_staging': (
            None
            if getattr(node, '_carried_staging_solution', None) is None
            else node._carried_staging_solution.copy()
        ),
        'post_retreat_shelf_front_x': getattr(
            node,
            '_post_retreat_shelf_front_x',
            None,
        ),
        'gravity_supported_payload': getattr(
            node,
            '_gravity_supported_payload',
            False,
        ),
        'supported_post_retreat_staging_required': getattr(
            node,
            '_supported_post_retreat_staging_required',
            False,
        ),
    }


@pytest.mark.parametrize('rejection', ['opening', 'edge', 'plan_transition', 'candidate', 'fresh_1'])
def test_empty_pickup_protocol_rejection_stops_before_first_command(rejection):
    commands = []
    def configure(node):
        node._open_gripper = lambda: commands.append('open') or True
        node._move_torso = lambda *args, **kwargs: commands.append('torso') or True
        node._move_arm_solution = lambda *args: commands.append('arm') or True
        node._close_for_grasp = lambda: pytest.fail('closure followed a rejected empty preflight')
    with pytest.raises(RuntimeError, match='Empty pickup|fixture_empty_candidate_rejected|fixture_fresh_1'):
        _mocked_pick_trace([.70, 0., 1.58], configure_node=configure, empty_guard_failure=rejection)
    assert commands == []


def test_empty_pickup_final_freshness_rejection_prevents_closure_and_loaded_motion():
    commands = []
    def configure(node):
        node._open_gripper = lambda: commands.append('open') or True
        node._move_torso = lambda *args, **kwargs: commands.append('torso') or True
        node._move_arm_solution = lambda q, duration: commands.append(('arm', int(q[1]))) or True
        node._close_for_grasp = lambda: pytest.fail('closure followed the final freshness rejection')
        node._recover_closed_pick = lambda **kw: pytest.fail('recovery followed an empty-stage rejection')
    with pytest.raises(RuntimeError, match='fixture_fresh_6'):
        _mocked_pick_trace([.70, 0., 1.58], configure_node=configure, empty_guard_failure='fresh_6')
    assert commands == ['open', 'torso', ('arm', 10), ('arm', 11), ('arm', 12)]


def test_empty_pickup_workflow_checks_all_expected_endpoints_and_open_admission():
    trace = _mocked_pick_trace([.70, 0., 1.58])
    calls = trace['empty_geometry_calls']
    assert [c[0] for c in calls[:5]] == ['capture', 'opening', 'edge', 'plan_transition', 'candidate']
    fresh = [c for c in calls if c[0].startswith('fresh_')]
    assert [c[0] for c in fresh] == ['fresh_1', 'fresh_2', 'fresh_3', 'fresh_4', 'fresh_5', 'fresh_6']
    expected = [np.zeros(8), np.zeros(8), np.r_[.35, np.zeros(7)],
                np.full(8, 10.), np.full(8, 11.), np.full(8, 12.)]
    for actual, q in zip(fresh, expected):
        np.testing.assert_array_equal(actual[1], q)
    assert [c[2] for c in fresh] == [.018, .069, .069, .069, .069, .069]
    assert trace['gripper_admission'][0] == (.069, True)
    waits = [c for c in calls if c[0].startswith('wait_')]
    assert [c[0] for c in waits] == [
        'wait_empty_setup_torso', 'wait_empty_setup_clearance',
        'wait_empty_cartesian_approach', 'wait_empty_cartesian_approach',
    ]


def test_empty_pickup_endpoint_wait_rejection_prevents_first_arm_command():
    commands = []
    def configure(node):
        node._open_gripper = lambda: commands.append('open') or True
        node._move_torso = lambda *args, **kwargs: commands.append('torso') or True
        node._move_arm_solution = lambda *args: commands.append('arm') or True
    with pytest.raises(RuntimeError, match='fixture_wait_empty_setup_torso'):
        _mocked_pick_trace([.70, 0., 1.58], configure_node=configure,
                           empty_guard_failure='wait_empty_setup_torso')
    assert commands == ['open', 'torso']


@pytest.mark.parametrize('height', [1.25, 1.58])
def test_pick_never_recovers_or_reopens_after_unconfirmed_retained_stop(height):
    gripper_commands = []
    arm_moves = []
    loaded_attempts = []

    def configure(node):
        command_gripper = node._command_gripper
        move_arm = node._move_arm_solution
        open_gripper = node._open_gripper

        def gripper(position, *, respect_cancel=False):
            gripper_commands.append(position)
            return command_gripper(position, respect_cancel=respect_cancel)

        def arm(solution, duration):
            arm_moves.append(int(np.asarray(solution)[1]))
            return move_arm(solution, duration)

        def unresolved(goal, duration, legs, command, **kwargs):
            loaded_attempts.append((command, legs[0][2]))
            raise manipulation_node.RetainedMotionNotStopped('stop unconfirmed')

        def guarded_open():
            assert not loaded_attempts, 'unconfirmed stop reopened hand'
            return open_gripper()

        node._command_gripper = gripper
        node._move_arm_solution = arm
        node._send_retained_arm_trajectory = unresolved
        node._recover_closed_pick = lambda **kwargs: pytest.fail(
            'recovery started while an accepted arm action might still move')
        node._open_gripper = guarded_open

    with pytest.raises(manipulation_node.RetainedMotionNotStopped, match='stop unconfirmed'):
        _mocked_pick_trace([.70, -.05, height], configure_node=configure)
    assert gripper_commands == [pytest.approx(.069)]
    assert arm_moves == [10, 11, 12]
    assert loaded_attempts == [('pick', 'extraction')]


def _observe_mock_pick_preflight(events):
    def configure(node):
        publish = node._publish_status

        def status(event, **fields):
            events.append(('status', event))
            return publish(event, **fields)

        node._publish_status = status
        for name, label in (
            ('_plan_carried_return', 'carried_plan'),
            ('_command_gripper', 'gripper_motion'),
            ('_move_torso', 'torso_motion'),
            ('_move_arm_solution', 'arm_motion'),
        ):
            original = getattr(node, name)

            def record(*args, _original=original, _label=label, **kwargs):
                events.append((_label,))
                return _original(*args, **kwargs)

            setattr(node, name, record)
    return configure


def test_pick_open_book_guard_runs_after_planned_event_before_carried_plan_or_motion():
    events = []

    def check(node, front, solutions, *, torso_height):
        events.append(('open_book_guard',))
        np.testing.assert_allclose(front, [.70, -.05, 1.25])
        assert [int(q[1]) for q in solutions] == [10, 11, 12]
        assert torso_height == node.pick_torso_height == pytest.approx(.35)
        return SimpleNamespace(ok=True)

    trace = _mocked_pick_trace(
        [.70, -.05, 1.25], approach_guard=check,
        configure_node=_observe_mock_pick_preflight(events))
    assert trace['succeeded']
    planned = events.index(('status', 'pick_approach_planned'))
    guarded = events.index(('open_book_guard',))
    carried = events.index(('carried_plan',))
    first_motion = min(index for index, event in enumerate(events)
                       if event[0] in ('gripper_motion', 'torso_motion', 'arm_motion'))
    assert planned < guarded < carried < first_motion


@pytest.mark.parametrize('failure', ['intersection', 'exception'])
def test_pick_open_book_guard_failure_prevents_carried_plan_and_all_motion(failure):
    events = []

    def check(*args, **kwargs):
        events.append(('open_book_guard',))
        if failure == 'exception':
            raise RuntimeError('fixture geometry unavailable')
        return SimpleNamespace(
            ok=False, reason='open_approach_book_intersection',
            collision_link='gripper_left_base_link', leg_index=2, sample_index=17)

    expected = ('Open gripper approach rejected: open_approach_book_intersection; '
                'link=gripper_left_base_link, leg=2, sample=17'
                if failure == 'intersection' else 'fixture geometry unavailable')
    with pytest.raises(RuntimeError, match=expected):
        _mocked_pick_trace(
            [.70, -.05, 1.25], approach_guard=check,
            configure_node=_observe_mock_pick_preflight(events))
    assert events == [('status', 'pick_approach_planned'), ('open_book_guard',)]


@pytest.mark.parametrize(('field', 'value'), (
    ('approach_max_position_error_m', .000500001),
    ('approach_max_orientation_error_rad', .010000001),
))
def test_pick_rejects_inaccurate_fk_before_collision_planning_or_motion(field, value):
    events = []

    def configure(node):
        _observe_mock_pick_preflight(events)(node)

        def inaccurate(positions, rotation, solutions):
            events.append(('accuracy',))
            result = _sentinel_pick_accuracy(positions, rotation, solutions)
            result[field] = value
            return result

        node._pick_path_accuracy = inaccurate

    def unexpected_guard(*args, **kwargs):
        pytest.fail('Inaccurate IK reached the collision guard')

    with pytest.raises(RuntimeError, match='Pick Cartesian alignment exceeds precision tolerance'):
        _mocked_pick_trace(
            [.70, -.05, 1.25], configure_node=configure,
            approach_guard=unexpected_guard)
    assert events == [('accuracy',), ('status', 'pick_approach_planned')]


def test_pick_passes_dedicated_accuracy_tolerances_to_cartesian_solver():
    trace = _mocked_pick_trace([.70, -.05, 1.58])
    assert trace['solve']['position_tolerance'] == pytest.approx(.0005)
    assert trace['solve']['orientation_tolerance'] == pytest.approx(.01)
    planned = next(fields for event, fields in trace['status']
                   if event == 'pick_approach_planned')
    assert planned['approach_max_position_error_m'] == 0.
    assert planned['approach_max_orientation_error_rad'] == 0.
    assert planned['pick_position_tolerance_m'] == pytest.approx(.0005)
    assert planned['pick_orientation_tolerance_rad'] == pytest.approx(.01)


def _empty_gripper_trace_configuration(
    monkeypatch,
    events,
    *,
    enabled=True,
    failed_command=None,
    failure='return_false',
    rejected_check=None,
):
    geometry = import_module('erc_phase1_solution.shelf_cradle_geometry')

    def check_route(node, front, grasp, start, route, shelf_plane, *, aperture):
        phase = 'closed_setup' if aperture == 0.0 else 'open_approach'
        events.append(('check', phase))
        assert int(grasp[1]) == 12
        if phase == 'closed_setup':
            assert shelf_plane == pytest.approx(float(front[0]) - 0.065)
            assert start[0] == pytest.approx(0.35)
            assert [int(q[1]) for q in route] == [5, 6, 10]
        else:
            assert shelf_plane is None
            assert aperture == pytest.approx(0.069)
            assert int(start[1]) == 10
            assert [int(q[1]) for q in route] == [11, 12]
        return 'fixture_collision' if rejected_check == phase else None

    def check_opening(node, front, grasp, clearance, shelf_plane):
        events.append(('check', 'opening'))
        assert int(grasp[1]) == 12
        assert int(clearance[1]) == 10
        assert shelf_plane == pytest.approx(float(front[0]) - 0.065)
        return 'fixture_collision' if rejected_check == 'opening' else None

    monkeypatch.setattr(geometry, 'check_cradle_tool_route', check_route)
    monkeypatch.setattr(geometry, 'check_gripper_opening', check_opening)

    def configure(node):
        node.shelf_side_cradle_enabled = enabled
        node._current_seed = lambda: np.zeros(8)
        solve = node._solve_cartesian_path

        def solve_with_setup(*args, **kwargs):
            solutions, index, score, unused = solve(*args, **kwargs)
            return solutions, index, score, [np.full(8, 5.0), np.full(8, 6.0)]

        node._solve_cartesian_path = solve_with_setup
        plan_carried_return = node._plan_carried_return
        node._plan_carried_return = lambda *args, **kwargs: (
            events.append(('plan', None)) or plan_carried_return(*args, **kwargs)
        )
        command_gripper = node._command_gripper
        command_count = 0

        def gripper(position, *, respect_cancel=False):
            nonlocal command_count
            command_count += 1
            events.append(('gripper', position))
            if command_count == failed_command:
                if failure == 'raise':
                    raise RuntimeError('fixture gripper controller failure')
                return False
            return command_gripper(position, respect_cancel=respect_cancel)

        node._command_gripper = gripper
        move_torso = node._move_torso
        node._move_torso = lambda height, duration, **kwargs: (
            events.append(('torso', height)) or move_torso(height, duration)
        )
        move_arm = node._move_arm_solution
        node._move_arm_solution = lambda solution, duration: (
            events.append(('arm', int(solution[1]))) or move_arm(solution, duration)
        )
        close_for_grasp = node._close_for_grasp
        node._close_for_grasp = lambda: (
            events.append(('grasp', None)) or close_for_grasp()
        )

    return configure


def test_candidate_empty_gripper_opens_only_after_closed_setup(monkeypatch):
    events = []
    trace = _mocked_pick_trace(
        [0.70, -0.05, 1.58],
        configure_node=_empty_gripper_trace_configuration(monkeypatch, events),
    )

    assert trace['succeeded']
    assert events[:14] == [
        ('check', 'closed_setup'), ('check', 'opening'), ('check', 'open_approach'),
        ('plan', None),
        ('gripper', 0.069), ('gripper', 0.0), ('torso', 0.35),
        ('arm', 5), ('arm', 6), ('arm', 10), ('gripper', 0.069),
        ('arm', 11), ('arm', 12), ('grasp', None),
    ]
    assert trace['gripper_commands'] == [0.069, 0.0, 0.069]


@pytest.mark.parametrize('failure', ('return_false', 'raise'))
@pytest.mark.parametrize('failed_command', (2, 3))
def test_candidate_empty_gripper_failure_halts_before_next_motion(
    monkeypatch, failed_command, failure,
):
    events = []
    configure = _empty_gripper_trace_configuration(
        monkeypatch, events, failed_command=failed_command, failure=failure,
    )
    with pytest.raises(RuntimeError, match='^pick_recovery_failed$'):
        _mocked_pick_trace([0.70, -0.05, 1.58], configure_node=configure)

    expected = [('gripper', 0.069), ('gripper', 0.0)]
    if failed_command == 3:
        expected += [
            ('torso', 0.35), ('arm', 5), ('arm', 6), ('arm', 10),
            ('gripper', 0.069),
        ]
    assert [event for event in events if event[0] not in ('check', 'plan')] == expected


@pytest.mark.parametrize('rejected_check', ('closed_setup', 'opening', 'open_approach'))
def test_candidate_empty_gripper_collision_rejection_prevents_all_commands(
    monkeypatch, rejected_check,
):
    events = []
    configure = _empty_gripper_trace_configuration(
        monkeypatch, events, rejected_check=rejected_check,
    )
    with pytest.raises(RuntimeError, match='rejected: fixture_collision'):
        _mocked_pick_trace([0.70, -0.05, 1.58], configure_node=configure)

    phases = ['closed_setup', 'opening', 'open_approach']
    assert events == [('check', phase) for phase in phases[:phases.index(rejected_check) + 1]]


@pytest.mark.parametrize(('enabled', 'height'), ((False, 1.58), (True, 1.25)))
def test_candidate_empty_gripper_staging_requires_opt_in_and_top_row(
    monkeypatch, enabled, height,
):
    events = []
    trace = _mocked_pick_trace(
        [0.70, -0.05, height],
        configure_node=_empty_gripper_trace_configuration(
            monkeypatch, events, enabled=enabled,
        ),
    )
    assert trace['succeeded']
    assert trace['gripper_commands'] == [0.069]
    assert all(event[0] != 'check' for event in events)


def _prepared_empty_gripper_trace_configuration(
    monkeypatch, events, *, final_aperture=0., measurement_error=None,
    failed_command=None, failure='return_false',
):
    configure_candidate = _empty_gripper_trace_configuration(
        monkeypatch, events, failed_command=failed_command, failure=failure,
    )
    preparation = import_module('erc_phase1_solution.empty_arm_preparation')

    def configure(node):
        configure_candidate(node)
        node._empty_arm_staged = True
        node._empty_arm_contact_guard = True
        measured = {name: 0. for name in manipulation_node.IK_JOINTS}
        measured['gripper_left_finger_joint'] = 0.
        node._empty_arm_preparation = {
            'final_goal': [0., 0., 0.],
            'book_odom': [.70, -.05, 1.58],
            'prepared_joints': dict(measured),
        }
        calls = 0

        def check_empty_stationary(actual_node):
            nonlocal calls
            assert actual_node is node
            assert node._empty_arm_contact_guard is True
            calls += 1
            events.append(('empty_measurement', calls))
            if calls == 2:
                if measurement_error:
                    raise RuntimeError(measurement_error)
                measured['gripper_left_finger_joint'] = final_aperture
            return dict(measured), {'pose': [0., 0., 0.]}

        monkeypatch.setattr(preparation, 'check_empty_stationary', check_empty_stationary)

    return configure


def test_prepared_candidate_keeps_fingers_closed_until_new_checked_clearance(monkeypatch):
    events = []
    trace = _mocked_pick_trace(
        [.70, -.05, 1.58],
        configure_node=_prepared_empty_gripper_trace_configuration(monkeypatch, events),
    )

    assert trace['succeeded']
    assert trace['gripper_commands'] == [0., .069]
    assert [event for event in events if event[0] in ('gripper', 'torso', 'arm', 'grasp')][:9] == [
        ('gripper', 0.), ('torso', .35), ('arm', 5), ('arm', 6),
        ('arm', 10), ('gripper', .069), ('arm', 11), ('arm', 12), ('grasp', None),
    ]
    assert events.index(('plan', None)) < events.index(('empty_measurement', 2))
    assert events.index(('empty_measurement', 2)) < events.index(('gripper', 0.))


@pytest.mark.parametrize('kwargs,reason', [
    ({'final_aperture': .004001}, 'no longer measures closed empty'),
    ({'measurement_error': 'empty-arm joint measurement stale'}, 'joint measurement stale'),
])
def test_prepared_candidate_rechecks_empty_measurements_after_planning(
    monkeypatch, kwargs, reason,
):
    events = []
    with pytest.raises(RuntimeError, match=reason):
        _mocked_pick_trace(
            [.70, -.05, 1.58],
            configure_node=_prepared_empty_gripper_trace_configuration(
                monkeypatch, events, **kwargs,
            ),
        )
    assert ('plan', None) in events
    assert all(event[0] not in ('gripper', 'torso', 'arm', 'grasp') for event in events)


@pytest.mark.parametrize('failure', ('return_false', 'raise'))
@pytest.mark.parametrize('failed_command', (1, 2))
def test_prepared_candidate_gripper_failure_stops_before_next_motion(
    monkeypatch, failed_command, failure,
):
    events = []
    with pytest.raises(RuntimeError, match='^pick_recovery_failed$'):
        _mocked_pick_trace(
            [.70, -.05, 1.58],
            configure_node=_prepared_empty_gripper_trace_configuration(
                monkeypatch, events, failed_command=failed_command, failure=failure,
            ),
        )
    expected = [('gripper', 0.)]
    if failed_command == 2:
        expected += [
            ('torso', .35), ('arm', 5), ('arm', 6), ('arm', 10), ('gripper', .069),
        ]
    assert [event for event in events if event[0] in ('gripper', 'torso', 'arm', 'grasp')] == expected


def test_top_row_loaded_pick_stops_at_second_cartesian_solution():
    trace = _mocked_pick_trace([0.70, -0.05, 1.58])

    assert trace['succeeded']
    assert trace['solve']['endpoint_first']
    assert int(trace['plan']['clearance'][1]) == 11
    # The unloaded approach starts at sol0 (10), and loaded extraction stops
    # at sol1 (11).  No shelf-side q7 motion is allowed; extension, cradle,
    # lowering and compaction all remain cached until after the base retreat.
    assert trace['arm_moves'] == [10, 11, 12, 11]
    assert trace['arm_durations'][-1] == pytest.approx(5.8)
    assert trace['roll_guard_states'] == []
    assert trace['gripper_commands'] == [0.069]
    assert trace['retained_phases'] == ['extraction']
    assert trace['fresh_probes'] == []
    assert trace['settle_calls'] == pytest.approx([1.0, 0.20])
    assert trace['final_probe_states'] == [
        ('wait', True),
        ('wait', True),
        ('sample', True),
    ]
    assert not trace['retention_probe_active']
    assert not trace['payload_robot_watchdog_enabled']
    ik_ready = next(fields for event, fields in trace['status'] if event == 'ik_ready')
    assert ik_ready['grasp'][1] == pytest.approx(-0.051)
    assert ik_ready['grasp_lateral_offset'] == pytest.approx(-0.001)
    assert ik_ready['loaded_clearance_lift'] == pytest.approx(0.018)
    assert trace['solve']['positions'][0][2] == pytest.approx(1.583)
    assert trace['solve']['positions'][-1][2] == pytest.approx(1.565)
    assert ik_ready['lowering_waypoints'] == 0
    assert ik_ready['cradle_waypoints'] == 0
    assert ik_ready['cradle_roll_index'] is None
    assert ik_ready['post_retreat_compaction_required']
    assert trace['carried_staging'] == pytest.approx(np.full(8, 11.0))
    assert trace['torso_moves'] == [pytest.approx(0.35)]
    assert trace['post_retreat_shelf_front_x'] == pytest.approx(0.95)
    assert not trace['gravity_supported_payload']
    assert trace['supported_post_retreat_staging_required']


def test_top_row_empty_grasp_recovery_still_retracts_through_solution_zero():
    trace = _mocked_pick_trace(
        [0.70, -0.05, 1.58],
        grasp_verified=False,
    )

    assert not trace['succeeded']
    assert int(trace['plan']['clearance'][1]) == 11
    # No payload was retained, so reverse every Cartesian approach waypoint,
    # including sol0, before following the transition route to HOME.
    assert trace['arm_moves'] == [10, 11, 12, 11, 10]
    assert trace['home_follows'] == [tuple(manipulation_node.HOME[1:])]
    assert trace['gripper_commands'] == [0.069, 0.069]
    assert trace['torso_moves'] == [
        pytest.approx(0.35),
        pytest.approx(float(manipulation_node.HOME[0])),
    ]


def test_top_row_pick_does_not_dispatch_the_deferred_cradle():
    trace = _mocked_pick_trace(
        [0.70, -0.05, 1.58],
        cradle_failure='raise',
    )

    assert trace['succeeded']
    assert trace['arm_moves'] == [10, 11, 12, 11]
    assert trace['roll_guard_states'] == []
    assert trace['closed_recoveries'] == []
    assert not trace['gravity_supported_payload']
    assert trace['gripper_commands'] == [0.069]
    assert trace['torso_moves'] == [pytest.approx(0.35)]


@pytest.mark.parametrize(
    ('settle_succeeded', 'final_contact', 'expected_cause'),
    (
        (False, True, 'final_grasp_settle_failed'),
        (True, False, 'final_grasp_contact_lost'),
    ),
)
def test_top_row_pick_requires_fresh_contact_after_settle(
    settle_succeeded,
    final_contact,
    expected_cause,
):
    trace = _mocked_pick_trace(
        [0.70, -0.05, 1.58],
        settle_succeeded=settle_succeeded,
        final_contact=final_contact,
    )

    assert not trace['succeeded']
    expected_settle_calls = [pytest.approx(1.0)]
    if settle_succeeded:
        expected_settle_calls.append(pytest.approx(0.20))
    assert trace['settle_calls'] == expected_settle_calls
    assert len(trace['closed_recoveries']) == 1
    recovery = trace['closed_recoveries'][0]
    assert recovery['cause'] == expected_cause
    assert recovery['remaining_carried_legs'] == ()
    assert len(recovery['unloaded_recovery_route']) == 1
    np.testing.assert_allclose(
        recovery['unloaded_recovery_route'][0],
        manipulation_node.HOME,
    )
    assert trace['carried_staging'] is None
    assert trace['post_retreat_shelf_front_x'] is None


def test_lower_row_loaded_pick_still_retracts_through_solution_zero():
    trace = _mocked_pick_trace([0.70, -0.05, 1.25])

    assert trace['succeeded']
    assert not trace['solve']['endpoint_first']
    assert int(trace['plan']['clearance'][1]) == 10
    assert trace['arm_moves'] == [10, 11, 12, 11, 10, 20, 30]
    assert trace['gripper_commands'] == [0.069]
    ik_ready = next(fields for event, fields in trace['status'] if event == 'ik_ready')
    assert ik_ready['cradle_waypoints'] == 0
    assert not ik_ready['post_retreat_compaction_required']
    assert not trace['gravity_supported_payload']


@pytest.mark.parametrize('shutdown_error', (ExternalShutdownException, RuntimeError))
def test_mission_main_uses_file_only_fallback_after_external_shutdown(
    monkeypatch, shutdown_error
):
    node = SimpleNamespace(
        finished=False,
        aborted=[],
        finalized=[],
        destroyed=False,
    )
    node._abort = lambda reason: node.aborted.append(reason)
    node._finalize_without_ros = lambda reason: node.finalized.append(reason)
    node.destroy_node = lambda: setattr(node, 'destroyed', True)

    shutdown_calls = _invalidate_context(monkeypatch, mission_manager)
    monkeypatch.setattr(mission_manager, 'MissionManager', lambda: node)
    monkeypatch.setattr(
        mission_manager.rclpy,
        'spin',
        lambda unused: (_ for _ in ()).throw(shutdown_error()),
    )

    mission_manager.main()

    assert node.aborted == []
    assert node.finalized == ['shutdown']
    assert node.destroyed
    assert shutdown_calls == []


def test_file_only_fallback_suppresses_ros_event_logging():
    node = object.__new__(mission_manager.MissionManager)
    node.finished = False
    node.state = 'NAVIGATE'
    writes = []
    node._write_summary = lambda success, reason, *, log_event: writes.append(
        (success, reason, log_event)
    )

    node._finalize_without_ros('shutdown')
    node._finalize_without_ros('ignored_duplicate')

    assert node.finished
    assert node.state == 'ABORTED'
    assert writes == [(False, 'shutdown', False)]


def test_mission_main_falls_back_when_publishers_invalidate_first(monkeypatch):
    node = SimpleNamespace(
        finished=False,
        finalized=[],
        destroyed=False,
    )
    node._abort = lambda reason: (_ for _ in ()).throw(RuntimeError(reason))
    node._finalize_without_ros = lambda reason: node.finalized.append(reason)
    node.destroy_node = lambda: setattr(node, 'destroyed', True)

    shutdown_calls = []
    monkeypatch.setattr(mission_manager.rclpy, 'init', lambda args=None: None)
    monkeypatch.setattr(mission_manager.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(
        mission_manager.rclpy,
        'shutdown',
        lambda: shutdown_calls.append(True),
    )
    monkeypatch.setattr(mission_manager, 'MissionManager', lambda: node)
    monkeypatch.setattr(
        mission_manager.rclpy,
        'spin',
        lambda unused: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    mission_manager.main()

    assert node.finalized == ['shutdown']
    assert node.destroyed
    assert shutdown_calls == [True]


def test_close_book_mode_carries_confirmed_overview_row():
    node = object.__new__(mission_manager.MissionManager)
    node.mode_pub = object()
    node.target_column = 4
    node.target_colour = 'blue'
    node.row_confirmed = True
    node.detected_row = 3
    node.target_marker_odom = [2.755, 0., 2.26]
    node.shelf_normal = [-1., 0.]
    node._shelf_registration_stamp_ns = 1_000_000_000
    node._confirmed_book_point_odom = [2.82, .02, .935]
    node._confirmed_book_point_stamp_ns = 1_500_000_000
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=2_000_000_000))
    calls = []
    node._command = lambda publisher, mode, **fields: calls.append(
        (publisher, mode, fields)
    )

    node._perception_mode('books')

    assert calls[0][2].pop('book_selection_context')['confirmed_point_odom'] == [2.82, .02, .935]
    assert calls == [
        (
            node.mode_pub,
            'books',
            {
                'shelf_column_number': 4,
                'book_colour': 'blue',
                'confirmed_row': 3,
            },
        )
    ]


def test_overview_book_mode_does_not_invent_confirmed_row():
    node = object.__new__(mission_manager.MissionManager)
    node.mode_pub = object()
    node.target_column = 2
    node.target_colour = 'red'
    node.row_confirmed = False
    node.detected_row = None
    node.target_marker_odom = [2.755, 0., 2.26]
    node.shelf_normal = [-1., 0.]
    node._shelf_registration_stamp_ns = 1_000_000_000
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=2_000_000_000))
    calls = []
    node._command = lambda publisher, mode, **fields: calls.append(fields)

    node._perception_mode('books')

    assert calls[0].pop('book_selection_context')['confirmed_point_odom'] is None
    assert calls == [
        {'shelf_column_number': 2, 'book_colour': 'red'}
    ]


def test_goal_from_target_zero_bias_preserves_standoff_and_heading():
    node = object.__new__(mission_manager.MissionManager)
    target = np.asarray([3.0, 4.0, 1.2])
    normal = np.asarray([0.6, 0.8])

    implicit = node._goal_from_target(target, normal, 1.3)
    explicit = node._goal_from_target(target, normal, 1.3, 0.0)

    assert implicit == pytest.approx(explicit)
    assert implicit[:2] == pytest.approx(target[:2] + normal * 1.3)
    assert implicit[2] == pytest.approx(math.atan2(-normal[1], -normal[0]))


def test_goal_from_target_lateral_bias_moves_left_without_changing_heading():
    node = object.__new__(mission_manager.MissionManager)
    target = np.asarray([3.0, 2.0, 1.2])
    normal = np.asarray([1.0, 0.0])

    x, y, yaw = node._goal_from_target(target, normal, 0.65, 0.05)

    assert (x, y) == pytest.approx((3.65, 1.95))
    assert (math.cos(yaw), math.sin(yaw)) == pytest.approx((-1.0, 0.0))
    target_delta = target[:2] - np.asarray([x, y])
    local_y = -math.sin(yaw) * target_delta[0] + math.cos(yaw) * target_delta[1]
    assert local_y == pytest.approx(-0.05)


def test_find_book_applies_configured_lateral_bias_only_to_grasp_goal():
    node = object.__new__(mission_manager.MissionManager)
    node.finished = False
    node.wall_started = time.monotonic()
    node.trial_timeout = 1200.0
    node.state = 'FIND_BOOK'
    node.state_started = None
    node.perception_timeout = 18.0
    node.book_point = object()
    node.detected_row = 1
    node.row_confirmed = False
    node.shelf_normal = np.asarray([1.0, 0.0])
    node.grasp_standoff = 0.65
    node.grasp_lateral_bias = 0.05
    transformed_point = mission_manager.PointStamped()
    transformed_point.header.frame_id = 'odom'
    transformed_point.header.stamp.sec = 10
    transformed_point.point.x = 3.0
    transformed_point.point.y = 2.0
    transformed_point.point.z = 1.2
    node.tf_buffer = SimpleNamespace(
        transform=lambda *_args, **_kwargs: transformed_point
    )
    calls = []
    node._republish_score = lambda: calls.append(('score',))
    node._navigate = lambda x, y, yaw, purpose: calls.append(
        ('navigate', x, y, yaw, purpose)
    )
    node._set_state = lambda state, **fields: calls.append(
        ('state', state, fields)
    )
    node._elapsed_state = lambda: 0.0
    node._abort = lambda reason: (_ for _ in ()).throw(
        AssertionError(f'unexpected abort: {reason}')
    )

    node._tick()

    assert calls[0] == ('score',)
    navigation = calls[1]
    assert navigation[0] == 'navigate'
    assert navigation[1:3] == pytest.approx((3.65, 1.95))
    assert (math.cos(navigation[3]), math.sin(navigation[3])) == pytest.approx(
        (-1.0, 0.0)
    )
    assert navigation[4] == 'book_grasp_standoff'
    assert calls[2] == ('state', 'ALIGN_BOOK', {})
    assert node._confirmed_book_point_stamp_ns == 10_000_000_000
    assert node._confirmed_book_point_odom == pytest.approx((3.0, 2.0, 1.2))


def test_aligned_book_starts_reacquisition_without_arming_a_detached_profile():
    node = object.__new__(mission_manager.MissionManager)
    node.finished = False
    node.wall_started = time.monotonic()
    node.trial_timeout = 1200.0
    node.state = 'ALIGN_BOOK'
    node.book_point = object()
    node.detected_row = 1
    node.nav_command_pub = object()
    node._nav_reached = lambda: True
    calls = []
    node._perception_mode = lambda mode: calls.append(('mode', mode))
    node._command = lambda publisher, event, **fields: calls.append(
        ('nav_command', event, fields)
    )
    node._manipulate = lambda command: calls.append(('manipulate', command))
    node._set_state = lambda state, **fields: calls.append(
        ('state', state, fields)
    )

    node._tick()

    assert node.book_point is None
    assert calls == [
        ('mode', 'idle'),
        ('manipulate', 'look_book_row_1'),
        ('state', 'HEAD_REACQUIRE_BOOK', {'command': 'look_book_row_1'}),
    ]


def test_successful_pick_dispatches_carried_retreat_with_bound_profile():
    node = object.__new__(mission_manager.MissionManager)
    node.finished = False
    node.wall_started = time.monotonic()
    node.trial_timeout = 1200.0
    node.state = 'PICK'
    node.dry_run = False
    node.target_colour = 'red'
    node.target_physical_column = 3
    node.detected_row = 1
    node.target_book_model = 'book_col_3_row_2_red'
    node.robot_pose = (1.0, 2.0, 0.1)
    node.shelf_normal = np.asarray([1.0, 0.0])
    node.carried_shelf_retreat = 0.35
    node._manip_succeeded = lambda command: command == 'pick'
    node._manip_failed = lambda: False
    node._elapsed_state = lambda: 0.0
    calls = []
    node._perception_mode = lambda mode: calls.append(('mode', mode))
    node._navigate = lambda x, y, yaw, purpose, **kwargs: calls.append(
        ('navigate', x, y, yaw, purpose, kwargs)
    )
    node._set_state = lambda state, **fields: calls.append(
        ('state', state, fields)
    )
    node._abort = lambda reason: (_ for _ in ()).throw(
        AssertionError(f'unexpected abort: {reason}')
    )

    node._tick()

    assert calls == [
        ('mode', 'idle'),
        (
            'navigate',
            pytest.approx(1.35),
            pytest.approx(2.0),
            pytest.approx(0.1),
            'carried_shelf_retreat',
            {'profile': 'carried_retreat'},
        ),
        ('state', 'CLEAR_SHELF_WITH_BOOK', {}),
    ]


def test_successful_pick_without_exact_identity_aborts_before_retreat():
    node = object.__new__(mission_manager.MissionManager)
    node.finished = False
    node.wall_started = time.monotonic()
    node.trial_timeout = 1200.0
    node.state = 'PICK'
    node.dry_run = False
    node.target_colour = 'red'
    node.target_physical_column = 3
    node.detected_row = 1
    node.target_book_model = None
    node._manip_succeeded = lambda command: command == 'pick'
    node._manip_failed = lambda: False
    node._elapsed_state = lambda: 0.0
    calls = []
    node._abort = lambda reason: calls.append(('abort', reason))
    node._navigate = lambda *args, **kwargs: calls.append(('navigate', args, kwargs))

    node._tick()

    assert calls == [('abort', 'target_book_identity_not_confirmed')]


def test_perception_mode_uses_and_clears_confirmed_row_context():
    node = object.__new__(perception_node.PerceptionNode)
    from erc_phase1_solution.book_selection_context import make_context

    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)
    )
    node.mode = 'idle'
    node.target_column = 1
    node.target_colour = 'green'
    node.confirmed_book_row = None
    node.marker_history = [1]
    node.book_history = [1]
    node.bin_history = [1]
    node.bin_tracker = perception_node.BinTracker()
    node.bin_rgb_frames = []
    node.bin_depth_frames = []
    tracking_resets = []
    tracking_status = []
    node.target_tracker = SimpleNamespace(
        reset=lambda: tracking_resets.append(True)
    )
    node.last_tracking_depth_ns = 123
    node.last_tracking_status = 'previous'
    node._publish_tracking_status = lambda event, **fields: (
        tracking_status.append((event, fields))
    )
    status = []
    node._publish_status = lambda event, **fields: status.append((event, fields))

    books = perception_node.String()
    books.data = perception_node.encode_event(
        'books',
        shelf_column_number=5,
        book_colour='yellow',
        confirmed_row=4,
        book_selection_context=make_context(
            marker=[2.755, 0.0, 2.26], normal=[-1.0, 0.0],
            observed_ns=8_000_000_000, dispatch_ns=10_000_000_000,
            column=5, colour='yellow', confirmed_row=4,
            confirmed_point=[2.82, 0.02, 0.605],
            confirmed_stamp_ns=9_000_000_000,
        ),
    )
    node._on_mode(books)

    assert node.mode == 'books'
    assert node.target_column == 5
    assert node.target_colour == 'yellow'
    assert node.confirmed_book_row == 4
    assert node.book_selection_context.confirmed_row == 4
    assert node.book_selection_context.confirmed_point_stamp_ns == 9_000_000_000
    assert node.marker_history == []
    assert node.book_history == []
    assert node.bin_history == []
    assert tracking_resets == [True]
    assert tracking_status[-1][0] == 'target_tracking_unavailable'
    assert status[-1][0] == 'mode_changed'

    idle = perception_node.String()
    idle.data = perception_node.encode_event('idle')
    node._on_mode(idle)

    assert node.mode == 'idle'
    assert node.confirmed_book_row is None
    assert node.book_selection_context is None


@pytest.mark.parametrize(
    'event',
    (
        'ik_target',
        'ik_ready',
        'lower_shelf_pick_candidate_planned',
        'lower_shelf_pick_candidate_rejected',
        'lower_shelf_planning_stage',
        'empty_pickup_geometry_verified',
        'empty_pickup_endpoint_waiting',
        'empty_pickup_endpoint_verified',
        'empty_pickup_endpoint_rejected',
        'gripper_closed',
        'grasp_verified',
        'retention_verified',
        'motion_exception',
        'transport_compact',
    ),
)
def test_manipulation_milestones_are_logged_without_becoming_terminal(event):
    node = object.__new__(mission_manager.MissionManager)
    node.manip_event = {'event': 'succeeded', 'command': 'previous'}
    logs = []
    node._log = lambda name, **fields: logs.append((name, fields))
    message = mission_manager.String()
    message.data = mission_manager.encode_event(event, command='pick')

    node._on_manipulation_status(message)

    assert node.manip_event == {'event': 'succeeded', 'command': 'previous'}
    assert logs == [
        (
            'manipulation_event',
            {'payload': {'event': event, 'command': 'pick'}},
        )
    ]


def _fresh_bin_state_point():
    message = mission_manager.PointStamped()
    message.header.frame_id = 'camera_optical_frame'
    message.header.stamp.sec = 2
    message.point.z = 2.0
    return message


def _bin_state_node(state):
    node = object.__new__(mission_manager.MissionManager)
    node.state = state
    node.finished = False
    node.wall_started = time.monotonic()
    node.trial_timeout = 1200.0
    node.state_started = None
    node.perception_timeout = 18.0
    node.navigation_timeout = 50.0
    node.manipulation_timeout = 75.0
    node.bin_standoff = 0.72
    node.robot_pose = (0.0, 0.0, 0.0)
    node.bin_point = _fresh_bin_state_point()
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=2_100_000_000)
    )
    node.nav_event = None
    node.manip_event = None
    node.tf_buffer = SimpleNamespace(
        transform=lambda *_args, **_kwargs: SimpleNamespace(
            point=SimpleNamespace(x=-1.0, y=0.0, z=0.75)
        )
    )
    node._elapsed_state = lambda: 0.0
    node._abort = lambda reason: (_ for _ in ()).throw(
        AssertionError(f'unexpected abort: {reason}')
    )
    return node


def test_find_bin_navigates_to_robot_side_standoff_before_placement():
    node = _bin_state_node('FIND_BIN')
    calls = []
    node._log = lambda *args, **kwargs: calls.append(('log', args, kwargs))
    node._perception_mode = lambda mode: calls.append(('mode', mode))
    node._navigate = lambda x, y, yaw, purpose: calls.append(
        ('navigate', x, y, yaw, purpose)
    )
    node._set_state = lambda state, **fields: calls.append(
        ('state', state, fields)
    )

    node._tick()

    assert node.bin_point is None
    assert calls[0] == ('mode', 'idle')
    _, x, y, yaw, purpose = calls[1]
    assert (x, y) == pytest.approx((-0.28, 0.0))
    assert abs(math.sin(yaw)) < 1e-9
    assert math.cos(yaw) == pytest.approx(-1.0)
    assert purpose == 'bin_placement_standoff'
    assert calls[2] == ('state', 'ALIGN_BIN', {})


def test_bin_alignment_requires_a_fresh_reacquisition_before_place():
    node = _bin_state_node('ALIGN_BIN')
    calls = []
    node.nav_event = {'event': 'reached'}
    node._perception_mode = lambda mode: calls.append(('mode', mode))
    node._manipulate = lambda command: calls.append(('manipulate', command))
    node._set_state = lambda state, **fields: calls.append(
        ('state', state, fields)
    )

    node._tick()

    assert node.bin_point is None
    assert calls == [
        ('mode', 'idle'),
        ('manipulate', 'look_bin'),
        ('state', 'HEAD_REACQUIRE_BIN', {}),
    ]

    node.state = 'HEAD_REACQUIRE_BIN'
    node.manip_event = {'event': 'succeeded', 'command': 'look_bin'}
    calls.clear()
    node._tick()
    assert node.bin_point is None
    assert calls == [
        ('mode', 'bin'),
        ('state', 'REACQUIRE_BIN', {}),
    ]

    node.state = 'REACQUIRE_BIN'
    node.bin_point = _fresh_bin_state_point()
    calls.clear()
    node._tick()
    # Keep the producer active until PLACE acknowledges captured fresh input.
    assert calls == [
        ('manipulate', 'place'),
        ('state', 'PLACE', {}),
    ]
