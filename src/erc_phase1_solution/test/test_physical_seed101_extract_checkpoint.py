"""Focused pure checks for the strict held-book extraction checkpoint."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / 'live_physical_seed101_extract_checkpoint.py'
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location('physical_seed101_extract', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)

from erc_phase1_solution import (  # noqa: E402
    seed101_outward_scaleup_continue_certificate as continue_certificate,
)
from erc_phase1_solution import (  # noqa: E402
    seed101_outward_reverse_reseat_certificate as reseat_certificate,
)


def checkpoint_book() -> probe.EntityPose:
    return probe.EntityPose(
        probe.CHECKPOINT_BOOK_POSITION_WORLD_M,
        probe.CHECKPOINT_BOOK_QUATERNION_XYZW,
    )


def nominal_scene():
    return {
        name: probe.EntityPose(position, probe.NOMINAL_BOOK_QUATERNION)
        for name, position in probe.BOOK_LAYOUT.items()
    }


def reverse_reseat_poses():
    start = probe.EntityPose(
        reseat_certificate.CHECKPOINT_BOOK_POSITION_WORLD_M,
        reseat_certificate.CHECKPOINT_BOOK_QUATERNION_XYZW,
    )
    stable_reference = probe.EntityPose(
        reseat_certificate.STABLE_REFERENCE_BOOK_POSITION_WORLD_M,
        reseat_certificate.STABLE_REFERENCE_BOOK_QUATERNION_XYZW,
    )
    return start, stable_reference


def reverse_reseat_support_floor_world_z() -> float:
    return (
        min(
            float(corner[2])
            for corner in (
                continue_certificate.CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M
            )
        )
        - continue_certificate.CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M
    )


def reverse_reseat_gate(**changes):
    start, stable_reference = reverse_reseat_poses()
    request = {
        'planned_delta_world_m': reseat_certificate.PLANNED_WORLD_DELTA_M,
        'support_floor_world_z_m': reverse_reseat_support_floor_world_z(),
        'stable_reference': stable_reference,
    }
    request.update(changes)
    return probe.inward_reseat_progress_gate(
        start,
        stable_reference,
        **request,
    )


def test_loaded_certificate_is_digest_bound_and_has_lifted_245mm_endpoint():
    valid, metrics = probe.validate_certificate()
    assert valid
    assert metrics['route_rows'] == 55
    assert metrics['micro_lift_rows'] == 5
    assert metrics['outward_rows'] == 49
    assert metrics['minimum_audit_margin_m'] == pytest.approx(
        0.0020226386064607915
    )
    assert probe.MICRO_LIFT_TOTAL_M == pytest.approx(0.005)
    assert probe.OUTWARD_TOTAL_M == pytest.approx(0.245)
    assert probe.ROUTE_FIRST_CLEAR_STEP == 49
    assert probe.ROUTE_FINAL_SHELF_CLEARANCE_M == pytest.approx(
        0.026355635550713075
    )
    assert probe.LOADED_RECOVERY_Q8[-1] == pytest.approx((
        0.349999786476243,
        -0.1798191544203312,
        0.7977004870554982,
        0.13849359814364273,
        -1.9687121235690639,
        0.2658143279576978,
        1.7269307283755084,
        0.21711333903326963,
    ))


def test_every_loaded_leg_densifies_to_two_milliradians():
    samples = 1
    for start, end in zip(
        probe.LOADED_RECOVERY_Q8,
        probe.LOADED_RECOVERY_Q8[1:],
    ):
        previous = start
        dense = probe.dense_arm_waypoints(
            start,
            end,
            maximum_increment=probe.ROUTE_DENSE_MAXIMUM_INCREMENT_RAD,
        )
        for point in dense:
            assert point[0] == pytest.approx(start[0], abs=1e-12)
            assert max(
                abs(after - before)
                for before, after in zip(previous[1:], point[1:])
            ) <= probe.ROUTE_DENSE_MAXIMUM_INCREMENT_RAD + 1e-12
            previous = point
            samples += 1
        assert dense[-1] == pytest.approx(end, abs=1e-12)
    assert samples == probe.AUDIT_DENSE_SAMPLES


def test_resume_checkpoint_requires_exact_arm_gripper_base_and_book():
    accepted = probe.resume_checkpoint_gate(
        probe.LOADED_RECOVERY_Q8[0],
        probe.CHECKPOINT_GRIPPER_MASTER_M,
        probe.Pose2(*probe.CHECKPOINT_BASE_WORLD_XYYAW),
        checkpoint_book(),
    )
    assert accepted.ok

    moved_arm = list(probe.LOADED_RECOVERY_Q8[0])
    moved_arm[-1] += probe.CHECKPOINT_ARM_LIMIT_RAD + 1e-6
    assert probe.resume_checkpoint_gate(
        moved_arm,
        probe.CHECKPOINT_GRIPPER_MASTER_M,
        probe.Pose2(*probe.CHECKPOINT_BASE_WORLD_XYYAW),
        checkpoint_book(),
    ).reason == 'held_checkpoint_left_arm_mismatch'

    shifted_book = probe.EntityPose(
        (
            probe.CHECKPOINT_BOOK_POSITION_WORLD_M[0]
            + probe.CHECKPOINT_BOOK_POSITION_LIMIT_M
            + 1e-6,
            *probe.CHECKPOINT_BOOK_POSITION_WORLD_M[1:],
        ),
        probe.CHECKPOINT_BOOK_QUATERNION_XYZW,
    )
    assert probe.resume_checkpoint_gate(
        probe.LOADED_RECOVERY_Q8[0],
        probe.CHECKPOINT_GRIPPER_MASTER_M,
        probe.Pose2(*probe.CHECKPOINT_BASE_WORLD_XYYAW),
        shifted_book,
    ).reason == 'held_checkpoint_book_position_mismatch'


def test_resume_scene_keeps_all_non_target_seed_gates_but_not_nominal_target_gate():
    first = nominal_scene()
    second = nominal_scene()
    second[probe.BOOK] = checkpoint_book()
    assert probe.resume_seed101_scene_gate(first, second).ok

    other = next(name for name in second if name != probe.BOOK)
    position = second[other].position
    second[other] = probe.EntityPose(
        (position[0] + 0.011, *position[1:]),
        second[other].quaternion,
    )
    assert probe.resume_seed101_scene_gate(first, second).reason == (
        'non_target_position_outside_seed101_gate'
    )


def test_progress_gate_requires_five_lifts_then_every_outward_step():
    start = checkpoint_book()
    previous = start
    for step in range(1, len(probe.LOADED_RECOVERY_Q8)):
        route = probe.RECOVERY_ROUTE[step]
        observed = probe.EntityPose(
            (
                start.position[0]
                - float(route['outward_world_minus_x_m']),
                start.position[1],
                start.position[2] + float(route['lift_world_z_m']),
            ),
            start.quaternion,
        )
        result = probe.extraction_progress_gate(
            start,
            previous,
            observed,
            step,
        )
        assert result.ok
        previous = observed

    assert result.metrics['shelf_face_clearance_m'] == pytest.approx(
        probe.ROUTE_FINAL_SHELF_CLEARANCE_M,
    )
    assert result.metrics['shelf_face_clearance_m'] >= (
        probe.FINAL_SHELF_CLEARANCE_REQUIRED_M
    )
    final_result = probe.final_extraction_geometry_gate(start, previous)
    assert final_result.ok
    assert final_result.metrics['final_shelf_clearance_m'] == pytest.approx(
        probe.ROUTE_FINAL_SHELF_CLEARANCE_M,
    )
    assert final_result.metrics['final_floor_clearance_m'] >= 0.003

    under_lifted = probe.EntityPose(
        (
            start.position[0],
            start.position[1],
            start.position[2] + 0.0004,
        ),
        start.quaternion,
    )
    assert probe.extraction_progress_gate(
        start,
        start,
        under_lifted,
        1,
    ).reason == 'book_did_not_follow_micro_lift_step'

    lifted = probe.EntityPose(
        (
            start.position[0],
            start.position[1],
            start.position[2] + probe.MICRO_LIFT_TOTAL_M,
        ),
        start.quaternion,
    )
    short_pull = probe.EntityPose(
        (
            lifted.position[0] - 0.0044,
            lifted.position[1],
            lifted.position[2],
        ),
        lifted.quaternion,
    )
    assert probe.extraction_progress_gate(
        start,
        lifted,
        short_pull,
        probe.MICRO_LIFT_COUNT + 1,
    ).reason == 'book_did_not_follow_outward_step'


def test_diagonal_peel_gate_accepts_only_small_outward_supported_motion():
    start = probe.EntityPose(
        (
            2.893645821378262,
            -0.15316229419440927,
            1.5773555286545407,
        ),
        (
            -0.004258650506574088,
            0.7093019208353551,
            0.004250302763156808,
            0.7048791271711482,
        ),
    )
    planned = (-0.00075, 0.0, 0.00050)
    observed = probe.EntityPose(
        tuple(value + delta for value, delta in zip(start.position, planned)),
        start.quaternion,
    )
    support_floor = probe.book_minimum_world_z(start) + 0.00000126
    policy = {
        'min_book_center_outward_progress_m': 0.00025,
        'max_book_deepest_extent_increase_m': 0.00015,
        'max_book_from_hand_translation_change_m': 0.00050,
        'max_incremental_rotation_rad': 0.002,
        'min_abs_rotation_axis_dot_world_y': 0.95,
        'max_yaw_component_rad': 0.0015,
        'min_floor_signed_distance_m': -0.0001,
        'max_floor_signed_distance_m': 0.0005,
    }
    accepted = probe.diagonal_peel_progress_gate(
        start,
        observed,
        planned_delta_world_m=planned,
        support_floor_world_z_m=support_floor,
        policy=policy,
    )
    assert accepted.ok
    assert accepted.metrics['book_center_outward_progress_m'] == pytest.approx(
        0.00075
    )
    assert accepted.metrics['book_floor_signed_m'] == pytest.approx(0.00049874)

    no_outward = probe.EntityPose(
        (start.position[0], start.position[1], start.position[2] + 0.0005),
        start.quaternion,
    )
    rejected = probe.diagonal_peel_progress_gate(
        start,
        no_outward,
        planned_delta_world_m=planned,
        support_floor_world_z_m=support_floor,
        policy=policy,
    )
    assert rejected.reason == 'diagonal_peel_lacked_outward_progress'


def test_shelf_outward_gate_can_require_deep_edge_progress_without_axis_noise():
    start = checkpoint_book()
    planned = (-0.0005, 0.0, 0.0)
    observed = probe.EntityPose(
        (start.position[0] - 0.00048, *start.position[1:]),
        start.quaternion,
    )
    result = probe.diagonal_peel_progress_gate(
        start,
        observed,
        planned_delta_world_m=planned,
        support_floor_world_z_m=probe.book_minimum_world_z(start),
        policy={
            'min_book_center_outward_progress_m': 0.00025,
            'min_book_deepest_extent_outward_progress_m': 0.00020,
            'max_book_deepest_extent_increase_m': 0.00010,
            'max_book_from_hand_translation_change_m': 0.00050,
            'max_incremental_rotation_rad': 0.002,
            'max_yaw_component_rad': 0.0015,
            'min_floor_signed_distance_m': -0.0001,
            'max_floor_signed_distance_m': 0.0002,
        },
    )
    assert result.ok
    assert result.metrics['book_deepest_edge_regression_m'] == pytest.approx(
        -0.00048
    )


def test_reverse_reseat_gate_accepts_exact_certified_stable_return():
    result = reverse_reseat_gate()

    assert result.ok
    assert result.reason == 'reverse_reseat_step_verified'
    assert result.metrics['book_center_inward_progress_m'] == pytest.approx(
        0.0009373341463958518,
    )
    assert result.metrics[
        'book_deepest_edge_inward_progress_m'
    ] == pytest.approx(0.0009509399779119576)
    assert result.metrics['book_hand_translation_change_m'] == pytest.approx(
        0.00010275043336367988,
    )
    assert result.metrics['book_cross_track_motion_m'] == pytest.approx(
        0.00008142875627496395,
    )
    assert result.metrics['book_incremental_rotation_rad'] == pytest.approx(
        0.0009421080608281189,
    )
    assert result.metrics['book_stable_reference_rotation_rad'] == pytest.approx(
        0.0,
        abs=1e-15,
    )
    assert result.metrics[
        'book_stable_reference_yaw_component_rad'
    ] == pytest.approx(0.0, abs=1e-15)
    assert result.metrics['book_rotation_growth_from_start_rad'] < 0.0
    assert -0.0001 <= result.metrics['book_floor_signed_m'] <= 0.00025
    assert result.metrics['book_reference_depth_overshoot_m'] == pytest.approx(
        0.0,
        abs=1e-15,
    )
    assert result.metrics[
        'book_stable_reference_position_error_m'
    ] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    'planned',
    (
        (),
        (0.0, 0.0, 0.0),
        (-0.001, 0.0, 0.0),
        (0.001, 1e-6, 0.0),
        (0.001, 0.0, 1e-6),
        (math.nan, 0.0, 0.0),
    ),
)
def test_reverse_reseat_gate_rejects_non_inward_plan(planned):
    result = reverse_reseat_gate(planned_delta_world_m=planned)

    assert not result.ok
    assert result.reason == 'reverse_reseat_plan_invalid'


@pytest.mark.parametrize(
    ('policy_key', 'metric_key', 'limit_delta', 'reason'),
    (
        (
            'min_book_center_inward_progress_m',
            'book_center_inward_progress_m',
            1e-9,
            'reverse_reseat_lacked_inward_progress',
        ),
        (
            'min_book_deepest_edge_inward_progress_m',
            'book_deepest_edge_inward_progress_m',
            1e-9,
            'reverse_reseat_lacked_deep_edge_inward_progress',
        ),
        (
            'max_book_from_hand_translation_change_m',
            'book_hand_translation_change_m',
            -1e-9,
            'reverse_reseat_book_hand_translation_mismatch',
        ),
        (
            'max_cross_track_motion_m',
            'book_cross_track_motion_m',
            -1e-9,
            'reverse_reseat_cross_track_motion_exceeded_limit',
        ),
        (
            'max_reference_depth_overshoot_m',
            'book_reference_depth_overshoot_m',
            -1e-9,
            'reverse_reseat_exceeded_reference_depth',
        ),
        (
            'max_stable_reference_position_error_m',
            'book_stable_reference_position_error_m',
            -1e-9,
            'reverse_reseat_stable_reference_position_mismatch',
        ),
        (
            'max_incremental_rotation_rad',
            'book_incremental_rotation_rad',
            -1e-9,
            'reverse_reseat_incremental_rotation_exceeded_limit',
        ),
        (
            'max_absolute_rotation_to_stable_reference_rad',
            'book_stable_reference_rotation_rad',
            -1e-9,
            'reverse_reseat_absolute_rotation_exceeded_limit',
        ),
        (
            'max_absolute_yaw_to_stable_reference_rad',
            'book_stable_reference_yaw_component_rad',
            -1e-9,
            'reverse_reseat_absolute_yaw_exceeded_limit',
        ),
        (
            'max_absolute_rotation_growth_from_start_rad',
            'book_rotation_growth_from_start_rad',
            -1e-9,
            'reverse_reseat_rotation_growth_exceeded_limit',
        ),
        (
            'min_floor_signed_distance_m',
            'book_floor_signed_m',
            1e-9,
            'reverse_reseat_shelf_penetration_exceeded_limit',
        ),
        (
            'max_floor_signed_distance_m',
            'book_floor_signed_m',
            -1e-9,
            'reverse_reseat_lost_shelf_support',
        ),
    ),
)
def test_reverse_reseat_gate_fails_closed_for_each_policy_limit(
    policy_key,
    metric_key,
    limit_delta,
    reason,
):
    accepted = reverse_reseat_gate()
    assert accepted.ok

    policy = {
        policy_key: accepted.metrics[metric_key] + limit_delta,
    }
    rejected = reverse_reseat_gate(policy=policy)

    assert not rejected.ok
    assert rejected.reason == reason


def test_world_pause_gate_requires_one_explicit_boolean():
    assert probe.world_paused_gate('paused: true\n', True).ok
    assert probe.world_paused_gate('paused: false\n', False).ok
    assert probe.world_paused_gate(
        'sim_time {\n  sec: 12\n}\niterations: 99\n',
        False,
    ).ok
    assert probe.world_paused_gate('paused: false\n', True).reason == (
        'world_pause_state_mismatch'
    )
    assert probe.world_paused_gate('', True).reason == (
        'world_pause_state_unavailable'
    )


def test_world_control_requires_observed_state_confirmation(monkeypatch):
    monkeypatch.setattr(probe, 'set_world_paused', lambda paused: True)
    monkeypatch.setattr(
        probe,
        'read_world_stats_message',
        lambda: 'paused: true\n',
    )
    monkeypatch.setattr(probe.time, 'sleep', lambda _: None)
    assert probe.set_world_paused_confirmed(True, attempts=1)
    assert not probe.set_world_paused_confirmed(False, attempts=2)


def test_exact_scene_reader_retries_but_never_accepts_partial(monkeypatch):
    samples = iter(('partial', 'complete'))
    monkeypatch.setattr(probe, 'read_dynamic_pose_message', lambda: next(samples))
    monkeypatch.setattr(probe.time, 'sleep', lambda _: None)

    def books(message):
        if message != 'complete':
            raise RuntimeError('partial')
        return {'complete': object()}

    monkeypatch.setattr(probe, 'book_poses_from_dynamic_pose', books)
    monkeypatch.setattr(
        probe,
        'entity_pose_from_dynamic_pose',
        lambda message, name: object(),
    )
    assert probe.read_exact_seed101_dynamic_pose_message() == 'complete'


def test_launcher_is_left_arm_only_and_has_no_entity_pose_mutation():
    source = SCRIPT.read_text(encoding='utf-8')
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert '_open_gripper' not in called_attributes
    assert '_command_gripper' not in called_attributes
    assert '_move_head' not in called_attributes
    assert '_move_torso' not in called_attributes
    assert '_pick' not in called_attributes
    assert '_stow' not in called_attributes
    assert 'set_model_pose' not in source
    assert '/set_pose' not in source
    assert 'strict held extraction rejected inherited motion' in source
    assert 'node._make_retained_arm_trajectory_goal(legs)' in source
    assert 'node._send_retained_arm_trajectory(' in source
    assert 'node._wait_for_retained_endpoint(' in source
    assert 'node._fresh_retention_probe(' in source
    assert 'node._adaptive_pressure_evidence(' in source
    assert "f'payload_pressure:{identity_reason or evidence.reason}'" in source
    assert 'node.set_strict_payload_motion(' in source
    assert 'leg_world_delta,' in source
    assert "f'post_extraction_step_{index}'" in source
    assert "raise RuntimeError(f'payload_hazard:{hazard}')" in source
    assert 'node._held_book_corners = np.asarray(' in source
    assert 'node._payload_monitor_enabled = True' in source
    assert 'pause: {str(bool(paused)).lower()}' in source
    assert 'set_world_paused_confirmed(True)' in source
    assert "'gz.msgs.WorldControl'" in source
    assert "'strict_pressure_pick_shelf_clear_extraction'" in source
    assert "'strict_pressure_pick_diagonal_peel'" in source
    assert "'outward-scaleup'," in source
    assert "'outward-scaleup-continue'," in source
    assert "'outward-settle-resample'," in source
    assert "'strict_pressure_pick_shelf_outward_probe'" in source
    assert "'strict_pressure_pick_outward_scaleup'" in source
    assert "'strict_pressure_pick_outward_scaleup_continue'" in source
    assert "'strict_pressure_pick_outward_settle_resample'" in source
    assert 'active_pause_every_endpoint' in source
    assert 'diagonal_peel_progress_gate(' in source
    assert 'paused_geometry = final_extraction_geometry_gate(' in source
    assert 'paused_evidence.verified' in source
    assert 'node._gz_pose_node.unsubscribe(DYNAMIC_POSE_TOPIC)' in source
    assert 'gazebo_entity_pose_mutation_used=False' in source
    assert 'right_arm_commanded=False' in source
    assert 'head_commanded=False' in source


def test_settle_resample_path_has_no_motion_rows_and_reuses_old_reference():
    source = SCRIPT.read_text(encoding='utf-8')

    assert 'active_q8 = SETTLE_ROUTE_Q8' in source
    assert 'active_peel_delta = (0.0, 0.0, 0.0)' in source
    assert 'active_settle_reference = EntityPose(' in source
    assert 'active_settle_planned_delta = tuple(' in source
    assert 'if active_settle_reference is not None' in source
    assert 'settle_rotation_growth_exceeded_limit' in source
    assert 'settle_absolute_pressure_out_of_bounds' in source
    assert 'paused_settle_absolute_pressure_out_of_bounds' in source
    assert 'read_exact_seed101_dynamic_pose_message()' in source
    assert 'for index, target_q8 in enumerate(active_q8[1:], start=1)' in source
    assert 'left_arm_trajectory_commanded=bool(len(active_q8) > 1)' in source
    assert 'gripper_trajectory_commanded=False' in source


@pytest.mark.parametrize(
    ('pressure', 'accepted'),
    (
        ((0.6, 0.6), True),
        ((0.5, 0.6), True),
        ((1.0, 0.6), True),
        ((0.4, 0.6), False),
        ((0.6, 0.4), False),
        ((1.01, 0.6), False),
        ((0.6, 1.01), False),
        ((0.5, 0.5), False),
    ),
)
def test_settle_pressure_bounds_are_enforced_live_and_paused(pressure, accepted):
    # Exercise the actual scalar gate expressions without importing ROS or
    # executing the launcher. Both fingers, overload, and total force matter.
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
    assignments = {
        node.targets[0].id: node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ('forces_ok', 'paused_forces_ok')
    }
    assert set(assignments) == {'forces_ok', 'paused_forces_ok'}
    context = {
        '__builtins__': {}, 'bool': bool, 'float': float,
        'minimum_left': 0.5, 'minimum_right': 0.5,
        'active_maximum_force_n': (1.0, 1.0),
        'active_minimum_total_force_n': 1.1,
        'final_pressure': pressure,
        'paused_evidence': SimpleNamespace(left_force=pressure[0], right_force=pressure[1]),
    }
    for expression in assignments.values():
        assert eval(compile(ast.Expression(body=expression), str(SCRIPT), 'eval'), context) is accepted


def test_reverse_reseat_stage_uses_inward_motion_and_its_dedicated_gate():
    source = SCRIPT.read_text(encoding='utf-8')
    tree = ast.parse(source)
    assert "reverse_reseat = args.stage == 'outward-reverse-reseat'" in source
    assert "active_result_stage = 'strict_pressure_pick_outward_reverse_reseat'" in source
    assert 'seed101_outward_reverse_reseat_certificate import (' in source
    motion_checks = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == 'motion_phase'
        and isinstance(node.ops[0], ast.In)
        and isinstance(node.comparators[0], ast.Tuple)
    ]
    assert any(ast.literal_eval(check.comparators[0]) == (
        'diagonal_peel', 'shelf_outward', 'shelf_inward',
    ) for check in motion_checks)

    # Check the moving, intermediate-paused, final-dwell and terminal-paused
    # branches themselves, independently of whether earlier stages use if/elif.
    branches = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == 'reverse_reseat'):
            continue
        body = ast.Module(body=node.body, type_ignores=[])
        calls = [call for call in ast.walk(body)
                 if isinstance(call, ast.Call)
                 and isinstance(call.func, ast.Name)
                 and call.func.id == 'inward_reseat_progress_gate']
        if calls:
            assert len(calls) == 1
            branches.append((node, calls[0]))
    branches.sort(key=lambda item: item[0].lineno)
    assert [ast.unparse(call.args[1]) for _, call in branches] == [
        'observed_book', 'paused_endpoint_book', 'final_book', 'paused_book',
    ]
    parents = {child: parent for parent in ast.walk(tree)
               for child in ast.iter_child_nodes(parent)}
    for branch, call in branches:
        assert ast.unparse(call.args[0]) == 'start_book'
        keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        assert keywords['stable_reference'] == 'active_reseat_reference'
        assert keywords['support_floor_world_z_m'] == 'active_support_floor_world_z'
        branch_source = ast.unparse(ast.Module(body=branch.body, type_ignores=[]))
        assert 'active_reseat_reference is None' in branch_source
        assert 'raise RuntimeError' in branch_source
        assert 'diagonal_peel_progress_gate(' not in branch_source
        fallback = ast.unparse(ast.Module(body=branch.orelse, type_ignores=[]))
        assert 'diagonal_peel_progress_gate(' in fallback
        parent = parents[call]
        if isinstance(parent, ast.Call):
            assert ast.unparse(parent.func) == '_require'
        else:
            assert isinstance(parent, ast.Assign)
            result_name = ast.unparse(parent.targets[0])
            assert result_name in ('progress_result', 'final_geometry', 'paused_geometry')
            assert any(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == '_require' and len(node.args) == 1
                and ast.unparse(node.args[0]) == result_name
                and node.lineno > branch.end_lineno
                for node in ast.walk(tree)
            )



def test_reverse_reseat_result_reports_reseat_without_claiming_extraction():
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
    results = [node for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == '_emit' and node.args
               and isinstance(node.args[0], ast.Constant)
               and node.args[0].value == 'result'
               and any(kw.arg == 'stage' and ast.unparse(kw.value)
                       == 'active_result_stage' for kw in node.keywords)
               and any(kw.arg == 'lift_distance_m'
                       and isinstance(kw.value, ast.IfExp)
                       for kw in node.keywords)]
    assert len(results) == 1
    fields = {kw.arg: kw.value for kw in results[0].keywords}
    for field in ('lift_distance_m', 'extraction_distance_m'):
        expression = fields[field]
        assert isinstance(expression, ast.IfExp)
        assert ast.literal_eval(expression.body) == 0.0
        assert isinstance(expression.test, ast.BoolOp)
        assert isinstance(expression.test.op, ast.Or)
        assert {ast.unparse(value) for value in expression.test.values} == {
            'reverse_reseat', 'post_reseat_settle', 'halfmillimeter_midroute_settle',
        }
    assert ast.unparse(fields['reseat_distance_m']) == (
        'float(active_peel_delta[0]) if reverse_reseat else 0.0'
    )
    assert ast.unparse(fields['left_arm_trajectory_commanded']) == (
        'bool(len(active_q8) > 1)'
    )
    assert ast.literal_eval(fields['gripper_trajectory_commanded']) is False



def test_intermediate_resume_baselines_passive_joints_while_paused():
    source = SCRIPT.read_text(encoding='utf-8')

    quiesced = source.index('monitor_quiesced.wait(timeout=1.0)')
    paused = source.index('pause_or_raise()', quiesced)
    baseline = source.index(
        'passive_generations = _joint_generation_snapshot(',
        paused,
    )
    unpause = source.index('set_world_paused_confirmed(False)', baseline)
    fresh_wait = source.index('_wait_for_joint_update(', unpause)

    assert quiesced < paused < baseline < unpause < fresh_wait


def test_final_pause_quiesces_monitor_and_rechecks_serialized_passive_groups():
    source = SCRIPT.read_text(encoding='utf-8')
    final_geometry = source.index('        _require(final_geometry)')
    pre_pause_clear = source.index('        require_monitor_clear()', final_geometry)
    suspend = source.index('        monitor_suspended.set()', pre_pause_clear)
    quiesce = source.index('monitor_quiesced.wait(timeout=1.0)', suspend)
    disabled = source.index('        node._payload_monitor_enabled = False', quiesce)
    final_pause = source.index('        pause_or_raise()', disabled)
    post_pause_clear = source.index('        require_monitor_clear()', final_pause)
    serialized = source.index('node.paused_serialized_joint_positions(', post_pause_clear)
    paused_state_gate = source.index('if not paused_state_stats[2]:', serialized)
    right_arm_gate = source.index('paused_joint_snapshot(RIGHT_ARM_JOINTS)', paused_state_gate)
    right_gripper_gate = source.index('paused_joint_snapshot(RIGHT_GRIPPER_JOINTS)', right_arm_gate)
    head_gate = source.index('paused_joint_snapshot(HEAD_JOINTS)', right_gripper_gate)
    monitor_stop = source.index('        monitor_stop.set()', head_gate)
    monitor_join = source.index('            monitor_thread.join(timeout=0.25)', monitor_stop)
    assert (pre_pause_clear < suspend < quiesce < disabled < final_pause
            < post_pause_clear < serialized < paused_state_gate < right_arm_gate
            < right_gripper_gate < head_gate < monitor_stop < monitor_join)
    # Quiescing is a deliberate pause handshake. Failure must raise, and only
    # confirmed paused Gazebo state may replace the frozen ROS joint stream.
    assert 'raise RuntimeError(' in source[quiesce:disabled]
    assert 'raise RuntimeError(' in source[paused_state_gate:right_arm_gate]
    assert source[disabled:final_pause].strip() == 'node._payload_monitor_enabled = False'
    assert 'stationary_joint_gate(' in source[paused_state_gate:right_arm_gate]
    assert 'stationary_joint_gate(' in source[right_arm_gate:right_gripper_gate]
    assert 'stationary_joint_gate(' in source[right_gripper_gate:head_gate]
    assert 'limit=PASSIVE_GRIPPER_STATIONARY_LIMIT_M' in source[right_arm_gate:head_gate]
    pause_helper = source[source.index('    def pause_or_raise()'):source.index('    def latch_passive_fault(')]
    assert 'stop.publish_zero()' in pause_helper
    assert 'if not set_world_paused_confirmed(True):' in pause_helper
    assert "raise RuntimeError('Gazebo world pause was not confirmed')" in pause_helper
