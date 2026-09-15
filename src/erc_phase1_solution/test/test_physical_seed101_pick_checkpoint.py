"""Focused pure tests for the strict physical seed-101 checkpoint."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import sys
import threading
import time

import pytest


SCRIPT = Path(__file__).parents[1] / 'live_physical_seed101_pick_checkpoint.py'
REPO_SRC = SCRIPT.parents[1]
SPEC = importlib.util.spec_from_file_location('physical_seed101_pick', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def nominal_scene():
    return {
        name: probe.EntityPose(position, probe.NOMINAL_BOOK_QUATERNION)
        for name, position in probe.BOOK_LAYOUT.items()
    }


def certified_source_assets():
    description = REPO_SRC / 'erc_description'
    return {
        'simulation_launch': REPO_SRC / 'erc_bringup' / 'launch' / 'simulation.launch.py',
        'trajectory_controller_params': REPO_SRC / 'erc_bringup' / 'config' / 'controller_params.yaml',
        'controller_manager_params': REPO_SRC / 'erc_bringup' / 'config' / 'gazebo_controller_manager_cfg.yaml',
        'gripper_command_clamp': REPO_SRC / 'erc_bringup' / 'scripts' / 'gripper_command_clamp.py',
        'world_sdf': description / 'worlds' / 'erc_world.sdf',
        'book_sdf': description / 'models' / 'book' / 'sdf' / 'erc_book.sdf',
        'shelf_sdf': description / 'models' / 'shelf' / 'sdf' / 'erc_shelf.sdf',
        'shelf_mesh': description / 'models' / 'shelf' / 'meshes' / 'erc_base_shelf.STL',
        'table_sdf': description / 'models' / 'table' / 'sdf' / 'erc_table.sdf',
        'table_mesh': description / 'models' / 'table' / 'meshes' / 'erc_base_table.STL',
        'collection_bin_sdf': description / 'models' / 'collection_bin' / 'sdf' / 'erc_collection_bin.sdf',
        'collection_bin_mesh': description / 'models' / 'collection_bin' / 'meshes' / 'erc_base_collection_bin.STL',
        'robot_urdf': description / 'urdf' / 'tiago_pro.urdf',
    }


def historical_v1_0_3_source_assets():
    """Bind the old route to its actual book/robot, not current official physics."""
    assets = certified_source_assets()
    historical = Path(__file__).parent / 'data' / 'official_v1_0_3_certificate'
    assets['book_sdf'] = historical / 'erc_book.sdf'
    assets['robot_urdf'] = historical / 'tiago_pro.urdf'
    return assets


def robot_package_shares():
    return {
        'omni_base_description': REPO_SRC / 'omni_base_robot' / 'omni_base_description',
        'tiago_pro_description': REPO_SRC / 'tiago_pro_robot' / 'tiago_pro_description',
        'tiago_pro_head_description': REPO_SRC / 'tiago_pro_head_robot' / 'tiago_pro_head_description',
        'pal_sea_arm_description': REPO_SRC / 'pal_sea_arm' / 'pal_sea_arm_description',
        'pal_pro_gripper_description': REPO_SRC / 'pal_pro_gripper' / 'pal_pro_gripper_description',
    }


def test_certified_base_route_and_exact_eight_q8_rows_are_pinned():
    assert probe.BASE_ROUTE_WORLD == (
        (1.3, -0.149392781, math.pi / 2.0),
        (1.3, -0.149392781, -math.pi / 4.0),
        (2.002948701, -0.149392781, -math.pi / 4.0),
    )
    assert len(probe.OBLIQUE_Q8) == 8
    assert probe.OBLIQUE_Q8[0][1:] == pytest.approx((
        -0.197657721832, 0.768592787207, 0.133023008060,
        -1.977213808290, 0.251840407735, 1.701923559526,
        0.224450639220,
    ))
    assert probe.OBLIQUE_Q8[-1][1:] == pytest.approx((
        -0.342517593645, 0.662142376208, 0.071729532802,
        -1.233345168443, 0.152946889496, 0.828621692895,
        0.224447061045,
    ))
    assert all(row[0] == pytest.approx(0.35) for row in probe.OBLIQUE_Q8)
    assert probe.PRESSURE_CLOSE_Q8[:-1] == probe.OBLIQUE_Q8[:6]
    assert probe.PRESSURE_CLOSE_Q8[-1] == probe.PRESSURE_CLOSE_CENTER_Q8
    assert max(
        abs(after - before)
        for before, after in zip(
            probe.OBLIQUE_Q8[5][1:], probe.PRESSURE_CLOSE_CENTER_Q8[1:]
        )
    ) < 0.01
    assert probe.GRASP_LINK_TARGET_WORLD == pytest.approx(
        (2.81999684, -0.154194330902, 1.56882200)
    )
    assert probe.oblique_route_gate().ok
    assert not probe.oblique_route_gate(certificate_available=False).ok
    assert probe.seed101_layout_digest() == probe.LAYOUT_DIGEST


def test_historical_collision_certificate_is_bound_to_v1_0_3_assets():
    result = probe.certificate_asset_gate(
        historical_v1_0_3_source_assets(), robot_package_shares(),
    )
    assert result.ok
    assert result.metrics['asset_count'] == len(probe.CERTIFIED_ASSET_SHA256)
    assert result.metrics['robot_collision_bundle_file_count'] == 24


@pytest.mark.parametrize('changed_asset', ('book_sdf', 'robot_urdf'))
def test_historical_certificate_rejects_each_current_official_asset(changed_asset):
    # Replacing either asset independently must invalidate the old route.
    assets = historical_v1_0_3_source_assets()
    assets[changed_asset] = certified_source_assets()[changed_asset]
    result = probe.certificate_asset_gate(assets, robot_package_shares())
    assert not result.ok
    assert result.reason == f'certificate_asset_hash_mismatch:{changed_asset}'


def test_collision_certificate_rejects_one_byte_asset_change(tmp_path):
    assets = historical_v1_0_3_source_assets()
    changed_world = tmp_path / 'erc_world.sdf'
    changed_world.write_bytes(assets['world_sdf'].read_bytes() + b'\n<!--changed-->\n')
    assets['world_sdf'] = changed_world
    result = probe.certificate_asset_gate(assets, robot_package_shares())
    assert not result.ok
    assert result.reason == 'certificate_asset_hash_mismatch:world_sdf'


def test_text_asset_hash_is_line_ending_stable_but_binary_hash_is_not(tmp_path):
    lf = tmp_path / 'lf.txt'
    crlf = tmp_path / 'crlf.txt'
    lf.write_bytes(b'alpha\nbeta\n')
    crlf.write_bytes(b'alpha\r\nbeta\r\n')
    assert probe._file_sha256(lf, text=True) == probe._file_sha256(crlf, text=True)
    assert probe._file_sha256(lf) != probe._file_sha256(crlf)


def test_official_start_escape_is_named_and_fails_closed_without_certificate():
    accepted = probe.official_start_escape_gate((0.0,) * 8)
    settled = probe.official_start_escape_gate(
        (0.0, 0.0, 0.001, 0.0, -0.0061, -0.0037, -0.0050, 0.0)
    )
    unavailable = probe.official_start_escape_gate(
        (0.0,) * 8, certificate_available=False
    )
    resumed = probe.official_start_escape_gate(
        (0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    uncertified = probe.official_start_escape_gate(
        (0.0, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005)
    )

    assert accepted.ok
    assert settled.ok
    assert accepted.metrics['certificate_samples'] == 101
    assert unavailable.reason == 'official_start_escape_not_certified'
    assert not resumed.ok
    assert resumed.reason == 'left_arm_outside_certified_start_profile'
    assert uncertified.reason == 'left_arm_outside_certified_start_profile'


@pytest.mark.parametrize(
    ('start', 'end'),
    (
        (probe.START_TORSO_CLEAR_Q8, probe.HOME_Q8),
        ((0.10, 0.0, 0.001, 0.0, -0.0061, -0.0037, -0.0050, 0.0), probe.HOME_Q8),
        (probe.HOME_LIFT_Q8, probe.OBLIQUE_Q8[0]),
        *tuple(zip(probe.OBLIQUE_Q8, probe.OBLIQUE_Q8[1:])),
        (probe.OBLIQUE_Q8[5], probe.PRESSURE_CLOSE_CENTER_Q8),
    ),
)
def test_dense_controller_waypoints_end_exactly_and_never_exceed_10_milliradians(start, end):
    points = probe.dense_arm_waypoints(start, end)
    previous = start
    for point in points:
        assert point[0] == pytest.approx(start[0], abs=1e-12)
        assert max(abs(b - a) for a, b in zip(previous[1:], point[1:])) <= (
            probe.MAXIMUM_ARM_TRAJECTORY_INCREMENT_RAD + 1e-12
        )
        previous = point
    assert points[-1] == pytest.approx(end, abs=1e-12)


def test_torso_durations_stay_below_the_official_velocity_limit():
    startup_distance = (
        probe.START_TORSO_CLEAR_Q8[0] + probe.OFFICIAL_START_TORSO_LIMIT_M
    )
    lift_distance = probe.HOME_LIFT_Q8[0] - probe.HOME_Q8[0]
    assert startup_distance / probe.START_TORSO_ESCAPE_SECONDS < (
        probe.OFFICIAL_TORSO_MAXIMUM_VELOCITY_MPS
    )
    assert lift_distance / probe.HOME_LIFT_TORSO_SECONDS < (
        probe.OFFICIAL_TORSO_MAXIMUM_VELOCITY_MPS
    )


def test_exact_scene_gate_accepts_only_stable_seed101_layout():
    first = nominal_scene()
    second = nominal_scene()
    assert len(first) == 20
    assert probe.scene_gate(first, second).ok

    missing = dict(second)
    missing.pop(probe.BOOK)
    assert probe.scene_gate(first, missing).reason == 'seed101_book_set_mismatch'

    displaced = dict(second)
    target = displaced[probe.BOOK]
    displaced[probe.BOOK] = probe.EntityPose(
        (target.position[0] + 0.00051, *target.position[1:]),
        target.quaternion,
    )
    assert probe.scene_gate(first, displaced).reason == (
        'target_position_outside_seed101_gate'
    )


def test_scene_gate_rejects_target_motion_inside_absolute_pose_tolerance():
    first = nominal_scene()
    second = nominal_scene()
    target = second[probe.BOOK]
    # Both poses remain within 0.5 mm of nominal, but the fresh-dwell motion is
    # greater than the independent 0.25 mm stability limit.
    first[probe.BOOK] = probe.EntityPose(
        (target.position[0] - 0.00015, *target.position[1:]),
        target.quaternion,
    )
    second[probe.BOOK] = probe.EntityPose(
        (target.position[0] + 0.00015, *target.position[1:]),
        target.quaternion,
    )
    result = probe.scene_gate(first, second)
    assert not result.ok
    assert result.reason == 'target_not_stable_in_fresh_dwell'


def test_postclose_gate_requires_stable_pose_and_shelf_support(monkeypatch):
    nominal = nominal_scene()[probe.BOOK]
    first = probe.EntityPose(
        (nominal.position[0], nominal.position[1] + 0.0010, nominal.position[2]),
        nominal.quaternion,
    )
    second = probe.EntityPose(
        (nominal.position[0], nominal.position[1] + 0.0011, nominal.position[2]),
        nominal.quaternion,
    )
    accepted = probe.postclose_target_gate(nominal, first, second)
    assert accepted.ok
    assert accepted.metrics['postclose_minimum_shelf_overlap_m'] > 0.20

    shifted = probe.EntityPose(
        (nominal.position[0], nominal.position[1] + 0.006, nominal.position[2]),
        nominal.quaternion,
    )
    assert probe.postclose_target_gate(nominal, shifted, shifted).reason == (
        'target_shifted_during_close'
    )

    tilted = probe.EntityPose(nominal.position, (0.0, 0.0, 0.0, 1.0))
    assert probe.postclose_target_gate(nominal, tilted, tilted).reason == (
        'target_tilted_during_close'
    )

    lifted = probe.EntityPose(
        (nominal.position[0], nominal.position[1], nominal.position[2] + 0.004),
        nominal.quaternion,
    )
    assert probe.postclose_target_gate(nominal, lifted, lifted).reason == (
        'target_lost_shelf_top_support'
    )

    monkeypatch.setattr(
        probe,
        'POST_CLOSE_MINIMUM_SHELF_OVERLAP_M',
        accepted.metrics['postclose_minimum_shelf_overlap_m'] + 1e-6,
    )
    assert probe.postclose_target_gate(nominal, first, second).reason == (
        'target_lost_shelf_support_during_close'
    )


def test_joint_snapshot_rejects_stale_per_joint_feedback():
    class FakeNode:
        def __init__(self):
            self._lock = threading.Lock()
            self.joints = {'passive_joint': 0.0}
            self._strict_joint_state_wall_times = {
                'passive_joint': (
                    time.monotonic() - probe.JOINT_STATE_MAXIMUM_WALL_AGE_S - 0.01
                ),
            }

    with pytest.raises(RuntimeError, match='joint state stale'):
        probe._joint_snapshot(FakeNode(), ('passive_joint',))


def test_shelf_pose_is_part_of_the_exact_route_certificate():
    shelf = probe.EntityPose(probe.SHELF_POSITION, probe.SHELF_QUATERNION)
    assert probe.shelf_gate(shelf, shelf).ok

    shifted = probe.EntityPose(
        (probe.SHELF_POSITION[0] + probe.SHELF_POSITION_LIMIT_M + 1e-6,
         *probe.SHELF_POSITION[1:]),
        probe.SHELF_QUATERNION,
    )
    assert probe.shelf_gate(shelf, shifted).reason == (
        'shelf_position_outside_certificate'
    )


def test_passive_joint_gate_detects_right_or_head_drift():
    assert probe.CERTIFIED_START_LEFT_PROFILES[0] == (0.0,) * 7
    assert len(probe.CERTIFIED_SETTLED_START_PROFILES) == 4
    assert probe.OFFICIAL_PASSIVE_RIGHT_GRIPPER_Q == (0.0,)
    assert probe.OFFICIAL_SETTLED_PROFILE_TOLERANCE_RAD == pytest.approx(0.001)
    for left, right, head in probe.CERTIFIED_SETTLED_START_PROFILES:
        assert probe.certified_settled_start_gate(left, right, head).ok
    left, _, _ = probe.CERTIFIED_SETTLED_START_PROFILES[-1]
    _, right, head = probe.CERTIFIED_SETTLED_START_PROFILES[0]
    assert probe.certified_settled_start_gate(left, right, head).reason == (
        'outside_certified_settled_start_profile'
    )
    reference = (0.0,) * 7
    assert probe.stationary_joint_gate(reference, reference).ok
    moved = (*reference[:-1], probe.PASSIVE_JOINT_STATIONARY_LIMIT_RAD + 1e-6)
    assert probe.stationary_joint_gate(reference, moved).reason == (
        'passive_joint_moved'
    )


def test_world_goal_conversion_round_trips_calibration():
    assert probe.WORLD_WAYPOINT_MAXIMUM_ATTEMPTS == 3
    assert probe.WORLD_WAYPOINT_SETTLE_SIM_SECONDS == pytest.approx(0.10)
    assert probe.WORLD_CORRECTION_POSITION_ENVELOPE_M == pytest.approx(0.05)
    assert probe.WORLD_CORRECTION_YAW_ENVELOPE_RAD == pytest.approx(0.02)
    world_base = probe.Pose2(0.012, -0.018, math.pi / 2.0 + 0.031)
    odom_base = probe.Pose2(-0.004, 0.001, -0.002)
    world_goal = probe.Pose2(*probe.BASE_ROUTE_WORLD[-1])
    odom_goal = probe.world_goal_to_odom(world_goal, world_base, odom_base)
    reconstructed = probe.compose(probe.world_from_odom(world_base, odom_base), odom_goal)
    assert reconstructed.x == pytest.approx(world_goal.x, abs=1e-12)
    assert reconstructed.y == pytest.approx(world_goal.y, abs=1e-12)
    assert probe.normalize_angle(reconstructed.yaw - world_goal.yaw) == pytest.approx(
        0.0, abs=1e-12
    )


def test_launcher_has_no_forbidden_motion_or_simulator_mutation_path():
    source = SCRIPT.read_text(encoding='utf-8')
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert 'live_pick_retreat_probe' not in source
    assert '_pick' not in called_attributes
    assert '_stow' not in called_attributes
    assert '_move_head' not in called_attributes
    assert '_attached_book_corners' not in called_attributes
    assert not any(name.startswith('_recover') for name in called_attributes)
    assert 'set_model_pose' not in source
    assert '/set_pose' not in source
    assert 'RIGHT_HOME' not in source
    assert "strict checkpoint rejected a right-arm/head command" in source
    assert 'class StrictLeftNode(ManipulationNode)' in source
    assert 'node._open_gripper()' in source
    assert 'node._adaptive_close_for_grasp()' in source
    assert 'node._fresh_retention_probe(' in source
    assert 'node.follow_dense_left(HOME_LIFT_Q8, OBLIQUE_Q8[0], 4.5)' in source
    assert 'node.follow_dense_left(measured_torso_clear_q8, HOME_Q8, 4.5)' in source
    assert 'measured_start_q8 = _joint_snapshot(node, IK_JOINTS)' in source
    assert 'measured_torso_clear_q8 = (' in source
    assert 'settled_start = certified_settled_start_gate(' in source
    assert "failure_reason='left_arm_changed_certified_start_profile'" in source
    assert 'for index, target in enumerate(PRESSURE_CLOSE_Q8[1:], start=1)' in source
    assert "insertion_scene_guard(f'after_insertion_leg_{index}')" in source
    assert 'from gz.transport13 import Node as GzTransportNode' in source
    assert 'from gz.msgs10.pose_v_pb2 import Pose_V' in source
    assert 'def arm_target_pose_watchdog(self) -> GateResult:' in source
    assert 'def target_pose_watchdog_reason(self) -> Optional[str]:' in source
    assert 'Pose_V, DYNAMIC_POSE_TOPIC, self._on_gz_dynamic_pose,' in source
    assert '_require(node.arm_target_pose_watchdog())' in source
    assert "require_target_pose_watchdog_clear('before_adaptive_close')" in source
    assert 'node.disarm_target_pose_watchdog()' in source
    assert "return 'target_moved_during_open_insertion'" in source
    assert "if command in ('cancel', 'abort', 'stop'):" in source
    assert 'self._strict_dispatch_lock = threading.Lock()' in source
    assert 'with self._strict_dispatch_lock:' in source
    assert 'super()._on_command(message)' in source
    assert 'strict checkpoint rejected an inherited trajectory path' in source
    assert 'def _execute_strict_trajectory(' in source
    assert 'def _wait_for_strict_endpoint(' in source
    assert 'previous = _joint_generation_snapshot(self, joints)' in source
    assert "failure_reason='strict_trajectory_endpoint_missed'" in source
    assert 'STRICT_ENDPOINT_SETTLE_WALL_TIMEOUT_S = 2.0' in source
    assert 'def move_torso_strict(' in source
    assert 'node.move_torso_strict(' in source
    assert 'START_TORSO_ESCAPE_SECONDS' in source
    assert 'HOME_LIFT_TORSO_SECONDS' in source
    assert 'node._move_torso(' not in source
    assert 'left_open_reference' not in source
    assert '(float(node.gripper_open),)' in source
    assert 'watch_target_contact and self.target_robot_contact_latched()' in source
    assert "require_not_cancelled('before_startup_gripper_open')" in source
    assert "require_not_cancelled('before_adaptive_close')" in source
    assert 'require_no_target_robot_contact(\'before_adaptive_close\')' in source
    assert "nav._finish_goal('cancelled', reason='checkpoint_cancel_requested')" in source
    navigate_source = source[
        source.index('    def navigate('):source.index('\n    try:', source.index('    def navigate('))
    ]
    assert 'for attempt in range(1, WORLD_WAYPOINT_MAXIMUM_ATTEMPTS + 1):' in navigate_source
    assert "read_dynamic_pose_message(), 'tiago_pro'," in navigate_source
    assert 'converted = world_goal_to_odom(goal, calibration_world, calibration_odom)' in navigate_source
    assert "'navigation_world_check'" in navigate_source
    assert "'navigation_correction_gate'" in navigate_source
    assert 'last_world_gate = final_base_gate(observed, goal)' in navigate_source
    assert "failure_reason='left_home_drifted_before_navigation'" in navigate_source
    assert "failure_reason='left_home_drifted_during_navigation'" in navigate_source
    assert 'position_error <= WORLD_CORRECTION_POSITION_ENVELOPE_M' in navigate_source
    assert 'yaw_error <= WORLD_CORRECTION_YAW_ENVELOPE_RAD' in navigate_source
    wait_index = navigate_source.index("while not nav.checkpoint_terminal.wait(0.02):")
    in_loop_cancel_index = navigate_source.index('if node._cancel.is_set():', wait_index)
    post_terminal_cancel_index = navigate_source.index(
        'if node._cancel.is_set():', in_loop_cancel_index + 1,
    )
    assert wait_index < in_loop_cancel_index < post_terminal_cancel_index
    assert 'nav._publish_zero()' in navigate_source[
        post_terminal_cancel_index:post_terminal_cancel_index + 180
    ]
    assert 'node.cancel_active_goals()' in source
    assert 'postclose_target_gate(preclose_target, first_final_book, final_book)' in source
    assert "require_no_target_robot_contact('final')" in source
    assert source.index('_require(certificate_asset_gate(asset_paths, package_shares))') < (
        source.index('rclpy.init(')
    )
    assert source.index("'left gripper startup open failed'") < source.index(
        "('side_translation', 'oblique_rotation', 'book_grasp_standoff')"
    )


def test_official_robot_model_has_no_wip_palm_observer():
    urdf = certified_source_assets()['robot_urdf'].read_text(encoding='utf-8')
    launch = certified_source_assets()['simulation_launch'].read_text(encoding='utf-8')
    assert 'arm_left_7_palm_contact' not in urdf
    assert 'WIP palm observer' not in launch
