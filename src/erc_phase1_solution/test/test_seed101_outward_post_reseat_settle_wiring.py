"""Focused pure-gate and static wiring checks for post-reseat settling."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / 'live_physical_seed101_extract_checkpoint.py'
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    'post_reseat_settle_wiring_probe',
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)

from erc_phase1_solution import (  # noqa: E402
    seed101_outward_post_reseat_settle_certificate as certificate,
)


SOURCE = SCRIPT.read_text(encoding='utf-8')


def post_reseat_poses():
    observed = probe.EntityPose(
        certificate.CHECKPOINT_BOOK_POSITION_WORLD_M,
        certificate.CHECKPOINT_BOOK_QUATERNION_XYZW,
    )
    stable_reference = probe.EntityPose(
        certificate.STABLE_REFERENCE_BOOK_POSITION_WORLD_M,
        certificate.STABLE_REFERENCE_BOOK_QUATERNION_XYZW,
    )
    return observed, stable_reference


def stable_gate(*, policy=None):
    observed, stable_reference = post_reseat_poses()
    return probe.stable_reseat_state_gate(
        observed,
        stable_reference=stable_reference,
        support_floor_world_z_m=certificate.SUPPORT_FLOOR_WORLD_Z_M,
        policy=policy,
    )


def permissive_policy():
    return {
        'max_stable_reference_position_error_m': 1.0,
        'max_stable_reference_maximum_x_error_m': 1.0,
        'max_absolute_rotation_to_stable_reference_rad': 1.0,
        'max_absolute_yaw_to_stable_reference_rad': 1.0,
        'min_floor_signed_distance_m': -1.0,
        'max_floor_signed_distance_m': 1.0,
    }


def test_stable_reseat_state_gate_accepts_bound_checkpoint_and_reports_metrics():
    result = stable_gate(policy={
        'max_stable_reference_position_error_m': 0.00025,
        'max_stable_reference_maximum_x_error_m': 0.00030,
        'max_absolute_rotation_to_stable_reference_rad': 0.00080,
        'max_absolute_yaw_to_stable_reference_rad': 0.00080,
        'min_floor_signed_distance_m': -0.00010,
        'max_floor_signed_distance_m': 0.00025,
    })

    assert result.ok
    assert result.reason == 'post_reseat_stable_state_verified'
    assert result.metrics == pytest.approx({
        'book_stable_reference_position_error_m': 0.00020608673087378054,
        'book_stable_reference_maximum_x_error_m': 0.00023998376269451782,
        'book_stable_reference_rotation_rad': 0.0003132162351817864,
        'book_stable_reference_yaw_component_rad': 0.00015708376298213892,
        'book_floor_signed_m': -6.796354348859168e-06,
        'book_maximum_world_x_m': 2.9725912773849252,
    }, abs=1e-15)


@pytest.mark.parametrize(
    ('policy_update', 'reason'),
    (
        (
            {'max_stable_reference_position_error_m': 0.00020},
            'post_reseat_stable_position_mismatch',
        ),
        (
            {'max_stable_reference_maximum_x_error_m': 0.00020},
            'post_reseat_stable_depth_mismatch',
        ),
        (
            {'max_absolute_rotation_to_stable_reference_rad': 0.00030},
            'post_reseat_absolute_rotation_exceeded_limit',
        ),
        (
            {'max_absolute_yaw_to_stable_reference_rad': 0.00015},
            'post_reseat_absolute_yaw_exceeded_limit',
        ),
        (
            {'min_floor_signed_distance_m': -0.000006},
            'post_reseat_shelf_penetration_exceeded_limit',
        ),
        (
            {'max_floor_signed_distance_m': -0.000007},
            'post_reseat_lost_shelf_support',
        ),
    ),
)
def test_stable_reseat_state_gate_fails_closed_for_each_limit(
    policy_update,
    reason,
):
    policy = permissive_policy()
    policy.update(policy_update)

    result = stable_gate(policy=policy)

    assert not result.ok
    assert result.reason == reason


def test_post_reseat_stage_binds_zero_command_route_and_stationary_policy():
    valid, metrics = certificate.validate_certificate()

    assert valid
    assert certificate.STAGE == 'outward-post-reseat-settle'
    assert certificate.ROUTE_Q8 == (certificate.CHECKPOINT_LEFT_Q8,)
    assert certificate.COMMAND_Q8 == ()
    assert metrics['route_rows'] == 1
    assert metrics['command_rows'] == 0

    stage_start = SOURCE.index(
        '    elif post_reseat_settle:\n'
        '        from erc_phase1_solution.'
        'seed101_outward_post_reseat_settle_certificate import ('
    )
    stage_end = SOURCE.index('    elif settle_resample:', stage_start)
    stage = SOURCE[stage_start:stage_end]

    assert 'active_route = POST_RESEAT_ROUTE' in stage
    assert 'active_q8 = POST_RESEAT_ROUTE_Q8' in stage
    assert 'active_peel_delta = (0.0, 0.0, 0.0)' in stage
    assert 'active_peel_duration = 0.0' in stage
    assert 'active_pause_every_endpoint = False' in stage
    assert 'active_minimum_force_retention_fraction = 0.0' in stage
    assert 'active_final_dwell_s = float(POST_RESEAT_DURATION_S)' in stage
    assert 'active_settle_reference = active_reseat_reference' in stage
    assert 'active_settle_rotation_growth_limit_rad = float(' in stage
    assert 'for index, target_q8 in enumerate(active_q8[1:], start=1):' in SOURCE


def test_post_reseat_split_stability_limits_are_imported_and_bound_distinctly():
    stage_start = SOURCE.index(
        '    elif post_reseat_settle:\n'
        '        from erc_phase1_solution.'
        'seed101_outward_post_reseat_settle_certificate import ('
    )
    stage_end = SOURCE.index('    elif settle_resample:', stage_start)
    stage = SOURCE[stage_start:stage_end]

    assert (
        'SETTLE_STABILITY_TRANSLATION_LIMIT_M as '
        'POST_RESEAT_STABILITY_TRANSLATION_LIMIT'
    ) in stage
    assert (
        'SETTLE_STABILITY_ROTATION_LIMIT_RAD as '
        'POST_RESEAT_STABILITY_ROTATION_LIMIT'
    ) in stage
    assert (
        'STARTUP_STABILITY_TRANSLATION_LIMIT_M as '
        'POST_RESEAT_STARTUP_TRANSLATION_LIMIT'
    ) in stage
    assert (
        'STARTUP_STABILITY_ROTATION_LIMIT_RAD as '
        'POST_RESEAT_STARTUP_ROTATION_LIMIT'
    ) in stage
    assert (
        'active_final_stability_translation_limit_m = float(\n'
        '            POST_RESEAT_STABILITY_TRANSLATION_LIMIT\n'
        '        )'
    ) in stage
    assert (
        'active_final_stability_rotation_limit_rad = float(\n'
        '            POST_RESEAT_STABILITY_ROTATION_LIMIT\n'
        '        )'
    ) in stage
    assert (
        'active_startup_stability_translation_limit_m = float(\n'
        '            POST_RESEAT_STARTUP_TRANSLATION_LIMIT\n'
        '        )'
    ) in stage
    assert (
        'active_startup_stability_rotation_limit_rad = float(\n'
        '            POST_RESEAT_STARTUP_ROTATION_LIMIT\n'
        '        )'
    ) in stage


def test_startup_and_final_held_stability_calls_use_their_own_limits():
    startup_start = SOURCE.index(
        '        _require(held_stability_gate(\n'
        '            first_book,\n'
        '            second_book,'
    )
    startup_end = SOURCE.index('\n        ))', startup_start) + len('\n        ))')
    startup = SOURCE[startup_start:startup_end]
    assert (
        'translation_limit_m=active_startup_stability_translation_limit_m'
        in startup
    )
    assert (
        'rotation_limit_rad=active_startup_stability_rotation_limit_rad'
        in startup
    )
    assert 'active_final_stability_' not in startup

    final_start = SOURCE.index(
        '        _require(held_stability_gate(\n'
        '            final_before_dwell,\n'
        '            final_book,'
    )
    final_end = SOURCE.index('\n        ))', final_start) + len('\n        ))')
    final = SOURCE[final_start:final_end]
    assert (
        'translation_limit_m=active_final_stability_translation_limit_m'
        in final
    )
    assert (
        'rotation_limit_rad=active_final_stability_rotation_limit_rad'
        in final
    )
    assert 'active_startup_stability_' not in final


def test_exact_checkpoint_gate_checks_live_and_serialized_snapshots():
    tree = ast.parse(SOURCE)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'active_checkpoint_gate'
    ]

    calls.sort(key=lambda call: call.lineno)
    assert [[ast.unparse(arg) for arg in call.args] for call in calls] == [
        [
            'tuple((serialized_joints[name] for name in IK_JOINTS))',
            'float(serialized_joints[LEFT_GRIPPER_JOINTS[0]])',
            'start_base_pose.planar',
            'start_book',
        ],
        [
            'tuple((final_joint_values[name] for name in IK_JOINTS))',
            'float(final_joint_values[LEFT_GRIPPER_JOINTS[0]])',
            'final_base_pose.planar',
            'final_book',
        ],
        ['arm', 'gripper', 'first_base', 'first_book'],
        [
            '_joint_snapshot(node, IK_JOINTS)',
            '_joint_snapshot(node, LEFT_GRIPPER_JOINTS)[0]',
            'second_base',
            'second_book',
        ],
        [
            '_joint_snapshot(node, IK_JOINTS)',
            'start_gripper',
            'start_base',
            'start_book',
        ],
    ]


def test_post_reseat_final_rotation_and_growth_limits_remain_tight():
    assert certificate.SETTLE_STABILITY_ROTATION_LIMIT_RAD == 0.000075
    assert certificate.SETTLE_ROTATION_GROWTH_LIMIT_RAD == 0.00005

    stage_start = SOURCE.index(
        '    elif post_reseat_settle:\n'
        '        from erc_phase1_solution.'
        'seed101_outward_post_reseat_settle_certificate import ('
    )
    stage_end = SOURCE.index('    elif settle_resample:', stage_start)
    stage = SOURCE[stage_start:stage_end]
    assert (
        'active_settle_rotation_growth_limit_rad = float(\n'
        '            POST_RESEAT_GROWTH_LIMIT\n'
        '        )'
    ) in stage


def test_post_reseat_uses_stable_gate_for_live_and_paused_endpoints():
    branches = []
    for branch in ast.walk(ast.parse(SOURCE)):
        if not (isinstance(branch, ast.If) and isinstance(branch.test, ast.Name)
                and branch.test.id == 'post_reseat_settle'):
            continue
        body = ast.Module(body=branch.body, type_ignores=[])
        assignments = [node for node in ast.walk(body)
                       if isinstance(node, ast.Assign)
                       and isinstance(node.value, ast.Call)
                       and isinstance(node.value.func, ast.Name)
                       and node.value.func.id == 'stable_reseat_state_gate']
        if assignments:
            assert len(assignments) == 1
            branches.append((branch, assignments[0]))
    branches.sort(key=lambda item: item[0].lineno)
    assert [ast.unparse(assignment.targets[0]) for _, assignment in branches] == [
        'final_geometry', 'paused_geometry',
    ]
    for (branch, assignment), expected_book in zip(branches, ('final_book', 'paused_book')):
        call = assignment.value
        assert [ast.unparse(arg) for arg in call.args] == [expected_book]
        keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        assert keywords['stable_reference'] == 'active_reseat_reference'
        assert keywords['policy'] == 'progress_policy(active_route[-1])'
        assert keywords['support_floor_world_z_m'] == 'active_support_floor_world_z'
        body = ast.unparse(ast.Module(body=branch.body, type_ignores=[]))
        assert 'active_reseat_reference is None' in body
        assert 'raise RuntimeError' in body
        assert 'expected_joint_gate(' in body
        assert 'IK_JOINTS' in body and 'LEFT_GRIPPER_JOINTS' in body



def test_post_reseat_pressure_is_fresh_exact_target_and_absolute():
    stage_start = SOURCE.index(
        '    elif post_reseat_settle:\n'
        '        from erc_phase1_solution.'
        'seed101_outward_post_reseat_settle_certificate import ('
    )
    stage_end = SOURCE.index('    elif settle_resample:', stage_start)
    stage = SOURCE[stage_start:stage_end]

    assert 'MINIMUM_FORCE_N as POST_RESEAT_MINIMUM_FORCE_N' in stage
    assert (
        'active_minimum_force_n = tuple(\n'
        '            float(value) for value in POST_RESEAT_MINIMUM_FORCE_N'
    ) in stage
    assert certificate.ENDPOINT_POLICY[
        'require_fresh_bilateral_exact_target_pressure'
    ] is True
    assert certificate.PRESSURE_REFERENCE['target_model'] == probe.BOOK
    assert certificate.MINIMUM_FORCE_N == pytest.approx((
        0.5 * certificate.PRESSURE_REFERENCE['left_force_n'],
        0.5 * certificate.PRESSURE_REFERENCE['right_force_n'],
    ))

    pressure_start = SOURCE.index(
        '    def pressure_evidence(stage: str) -> Tuple[float, float, float]:'
    )
    pressure_end = SOURCE.index(
        '\n    try:\n        _require(world_paused_gate(',
        pressure_start,
    )
    pressure = SOURCE[pressure_start:pressure_end]
    assert 'require_new_sample=True' in pressure
    assert 'evidence.verified' in pressure
    assert 'model == BOOK' in pressure
    assert 'identity_reason is None' in pressure
    assert 'absolute_force_ok' in pressure
    assert 'not node.target_robot_contact_latched()' in pressure
    pressure_stages = {
        node.args[0].value for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == 'pressure_evidence' and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert {'resume_transport_lock', 'post_extraction_stability'} <= pressure_stages


def test_post_reseat_continuous_watchdog_uses_stable_reference_orientation():
    stage_start = SOURCE.index(
        '    elif post_reseat_settle:\n'
        '        from erc_phase1_solution.'
        'seed101_outward_post_reseat_settle_certificate import ('
    )
    stage_end = SOURCE.index('    elif settle_resample:', stage_start)
    stage = SOURCE[stage_start:stage_end]

    assert (
        'active_continuous_stable_quaternion = tuple(\n'
        '            float(value) for value in '
        'POST_RESEAT_REFERENCE_QUATERNION'
    ) in stage
    assert (
        "POST_RESEAT_EXECUTION_POLICY[\n"
        "                'maximum_continuous_absolute_rotation_rad'"
    ) in stage
    assert (
        "POST_RESEAT_EXECUTION_POLICY[\n"
        "                'maximum_continuous_absolute_yaw_rad'"
    ) in stage
    assert (
        'node._strict_payload_stable_quaternion = (\n'
        '                tuple(active_continuous_stable_quaternion)'
    ) in SOURCE
    assert (
        'node._strict_payload_absolute_rotation_limit_rad = float('
    ) in SOURCE
    assert 'node._strict_payload_absolute_yaw_limit_rad = float(' in SOURCE
    assert 'absolute_rotation > absolute_rotation_limit' in SOURCE
    assert 'absolute_yaw > absolute_yaw_limit' in SOURCE


def test_post_reseat_reports_zero_motion_and_pauses_before_passive_final_checks():
    result_start = SOURCE.index("            'result',\n            passed=True,")
    result_end = SOURCE.index('    except Exception as exc:', result_start)
    result = SOURCE[result_start:result_end]

    # Evaluate only the scalar reporting expressions, without running the
    # launcher. Settling must not claim a lift, extraction, or reseat motion.
    tree = ast.parse(SOURCE)
    result_call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == '_emit' and node.args
        and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'result'
        and any(kw.arg == 'stage' and ast.unparse(kw.value)
                == 'active_result_stage' for kw in node.keywords)
        and any(kw.arg == 'lift_distance_m'
                and isinstance(kw.value, ast.IfExp) for kw in node.keywords)
    )
    fields = {kw.arg: kw.value for kw in result_call.keywords}
    for field in ('lift_distance_m', 'extraction_distance_m', 'reseat_distance_m'):
        expression = ast.Expression(body=fields[field])
        assert eval(compile(expression, str(SCRIPT), 'eval'), {
            '__builtins__': {}, 'reverse_reseat': False,
            'post_reseat_settle': True, 'halfmillimeter_midroute_settle': False,
        }) == 0.0
    assert (
        'float(active_peel_delta[0]) if reverse_reseat else 0.0'
    ) in result
    assert 'route_steps=len(active_q8) - 1' in result
    assert 'left_arm_trajectory_commanded=bool(len(active_q8) > 1)' in result
    assert 'gripper_trajectory_commanded=False' in result
    assert 'base_motion_commanded=False' in result
    assert 'right_arm_commanded=False' in result
    assert 'head_commanded=False' in result
    assert 'gazebo_entity_pose_mutation_used=False' in result
    assert 'gazebo_paused=True' in result

    final_start = SOURCE.index(
        "        final_pressure = pressure_evidence('post_extraction_stability')"
    )
    paused_gate = SOURCE.index(
        '            paused_geometry = stable_reseat_state_gate(',
        final_start,
    )
    final = SOURCE[final_start:paused_gate]
    pause = final.index('        pause_or_raise()')
    passive_right = final.index(
        '        _require(stationary_joint_gate(\n'
        '            right_reference,',
        pause,
    )
    passive_gripper = final.index(
        '        _require(stationary_joint_gate(\n'
        '            right_gripper_reference,',
        passive_right,
    )
    passive_head = final.index(
        '        _require(stationary_joint_gate(\n'
        '            head_reference,',
        passive_gripper,
    )
    monitor_stop = final.index('        monitor_stop.set()', passive_head)
    paused_pressure = final.index(
        '        paused_evidence, paused_model, paused_identity = (',
        monitor_stop,
    )
    paused_scene = final.index(
        '            _require(resume_seed101_scene_gate(first_books, paused_books))',
        paused_pressure,
    )
    assert pause < passive_right < passive_gripper < passive_head
    assert passive_head < monitor_stop < paused_pressure < paused_scene
