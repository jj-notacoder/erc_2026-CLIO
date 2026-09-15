"""Focused static checks for the pressure-refresh pose watchdog."""

from __future__ import annotations

import ast
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / 'live_physical_seed101_extract_checkpoint.py'
SOURCE = SCRIPT.read_text(encoding='utf-8')


def method_source(name: str, next_name: str) -> str:
    start = SOURCE.index(f'        def {name}(')
    end = SOURCE.index(f'\n        def {next_name}(', start)
    return SOURCE[start:end]


def pressure_evidence_source() -> str:
    start = SOURCE.index(
        '    def pressure_evidence(stage: str) -> Tuple[float, float, float]:'
    )
    end = SOURCE.index(
        '\n    try:\n        _require(world_paused_gate(',
        start,
    )
    return SOURCE[start:end]


def test_pose_watchdog_starts_only_for_an_enabled_payload_monitor():
    pressure = pressure_evidence_source()

    enabled = pressure.index(
        "        if bool(getattr(node, '_payload_monitor_enabled', False)):"
    )
    construct = pressure.index(
        '            pose_probe_thread = threading.Thread(',
        enabled,
    )
    target = pressure.index(
        '                target=pose_probe_watchdog,',
        construct,
    )
    start = pressure.index('            pose_probe_thread.start()', target)
    refresh = pressure.index('            retained = node._fresh_retention_probe(')

    assert enabled < construct < target < start < refresh


def test_pose_watchdog_spans_refresh_and_checks_synchronously_before_join():
    pressure = pressure_evidence_source()

    thread_start = pressure.index('            pose_probe_thread.start()')
    refresh_start = pressure.index(
        '            retained = node._fresh_retention_probe('
    )
    refresh_end = pressure.index('            if pose_probe_thread is not None', refresh_start)
    synchronous_check = pressure.index(
        '                reason = node.strict_pose_only_hazard_reason()',
        refresh_end,
    )
    finally_block = pressure.index('        finally:', synchronous_check)
    stop = pressure.index('            pose_probe_stop.set()', finally_block)
    join = pressure.index('                pose_probe_thread.join(timeout=0.25)', stop)
    fault_gate = pressure.index("        if pose_probe_fault['reason']:", join)

    assert thread_start < refresh_start < refresh_end < synchronous_check
    assert synchronous_check < finally_block < stop < join < fault_gate


def test_pose_watchdog_cancels_and_zeros_immediately_then_raises_after_join():
    pressure = pressure_evidence_source()

    watchdog_start = pressure.index('        def pose_probe_watchdog() -> None:')
    watchdog_end = pressure.index(
        "        if bool(getattr(node, '_payload_monitor_enabled', False)):",
        watchdog_start,
    )
    watchdog = pressure[watchdog_start:watchdog_end]
    assert 'reason = node.strict_pose_only_hazard_reason()' in watchdog
    watchdog_fault = watchdog.index('            if reason is not None:')
    watchdog_cancel = watchdog.index('                node._cancel.set()', watchdog_fault)
    watchdog_zero = watchdog.index('                stop.publish_zero()', watchdog_cancel)
    watchdog_return = watchdog.index('                return', watchdog_zero)
    assert watchdog_fault < watchdog_cancel < watchdog_zero < watchdog_return

    synchronous_start = pressure.index(
        '            if pose_probe_thread is not None '
        "and not pose_probe_fault['reason']:"
    )
    synchronous_end = pressure.index('        finally:', synchronous_start)
    synchronous = pressure[synchronous_start:synchronous_end]
    synchronous_fault = synchronous.index('            if reason is not None:')
    synchronous_cancel = synchronous.index(
        '                node._cancel.set()',
        synchronous_fault,
    )
    synchronous_zero = synchronous.index(
        '                stop.publish_zero()',
        synchronous_cancel,
    )
    assert synchronous_fault < synchronous_cancel < synchronous_zero

    join = pressure.index('                pose_probe_thread.join(timeout=0.25)')
    fault_gate = pressure.index("        if pose_probe_fault['reason']:", join)
    raise_fault = pressure.index('            raise RuntimeError(', fault_gate)
    fault_message = pressure.index(
        'payload pose hazard during pressure refresh:',
        raise_fault,
    )
    retention_gate = pressure.index('        if not retained:', fault_message)
    assert join < fault_gate < raise_fault < fault_message < retention_gate


def test_pose_only_check_keeps_every_non_contact_payload_safety_constraint():
    pose_only = method_source(
        'strict_pose_only_hazard_reason',
        'cancel_active_goals',
    )

    required_checks = (
        "if not bool(getattr(self, '_payload_monitor_enabled', False)):",
        'if self.target_robot_contact_latched():',
        "return hazard('payload_robot_contact')",
        'reference_target = self._strict_payload_target_reference',
        'reference_base = self._strict_payload_base_reference',
        'reference_gripper = self._strict_payload_gripper_reference',
        "gripper = self.joints.get('gripper_left_finger_joint')",
        "return hazard('payload_aperture_unavailable')",
        'aperture_error > PAYLOAD_APERTURE_DRIFT_LIMIT_M',
        "'payload_aperture_drift'",
        'self.strict_entity_pose_with_generation(BOOK)',
        "base = self.strict_entity_pose('tiago_pro').planar",
        "return hazard('payload_dynamic_pose_stale')",
        'translation_error = _distance(',
        'translation_error > float(translation_limit)',
        "'payload_pose_probe_translation_drift'",
        'rotation_error = quaternion_distance(',
        'rotation_error > float(rotation_limit)',
        "'payload_pose_probe_rotation_drift'",
        'absolute_rotation_vector = _world_rotation_vector(',
        'absolute_rotation > float(absolute_rotation_limit)',
        "'payload_absolute_rotation_exceeded_limit'",
        'absolute_yaw > float(absolute_yaw_limit)',
        "'payload_absolute_yaw_exceeded_limit'",
        'corner_displacement = (',
        'corner_displacement\n                '
        '> active_continuous_corner_displacement_limit_m',
        "'payload_continuous_corner_displacement'",
        'base.x - reference_base.x',
        'base.y - reference_base.y',
        '> PAYLOAD_BASE_POSITION_LIMIT_M',
        "return hazard('base_moved_during_loaded_extraction')",
        'math.sin(base.yaw - reference_base.yaw)',
        'math.cos(base.yaw - reference_base.yaw)',
        'yaw_error > PAYLOAD_BASE_YAW_LIMIT_RAD',
        "return hazard('base_rotated_during_loaded_extraction')",
    )
    for check in required_checks:
        assert check in pose_only


def test_pose_only_check_uses_wall_fresh_target_and_base_poses():
    pose_only = method_source(
        'strict_pose_only_hazard_reason',
        'cancel_active_goals',
    )
    strict_pose = method_source(
        'strict_entity_pose_with_generation',
        'strict_entity_pose',
    )

    assert 'self.strict_entity_pose_with_generation(BOOK)' in pose_only
    assert 'target_generation' in pose_only
    assert "base = self.strict_entity_pose('tiago_pro').planar" in pose_only
    assert 'maximum_wall_age_s: float = ' in strict_pose
    assert 'PAYLOAD_TARGET_POSE_MAXIMUM_WALL_AGE_S' in strict_pose
    assert 'time.monotonic_ns() - stamp_ns' in strict_pose
    assert 'if pose is None:' in strict_pose
    assert 'if age < 0.0 or age > float(maximum_wall_age_s):' in strict_pose
    assert "raise RuntimeError(f'Gazebo dynamic pose for {name!r} is stale')" in strict_pose


def test_pose_watchdog_corner_gate_uses_the_selected_stage_limit():
    tree = ast.parse(SOURCE)
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == 'strict_pose_only_hazard_reason')
    gates = [node for node in ast.walk(method)
             if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
             and node.left.id == 'corner_displacement']
    assert len(gates) == 1
    gate = gates[0]
    assert ast.unparse(gate) == (
        'corner_displacement > active_continuous_corner_displacement_limit_m'
    )
    code = compile(ast.Expression(body=gate), str(SCRIPT), 'eval')
    for limit in (0.0001, 0.00025):
        for displacement, rejected in ((limit / 2, False), (limit, False), (limit * 2, True)):
            assert eval(code, {
                '__builtins__': {}, 'corner_displacement': displacement,
                'active_continuous_corner_displacement_limit_m': limit,
            }) is rejected


def test_pose_only_check_omits_only_pressure_and_contact_loss_evidence():
    pose_only = method_source(
        'strict_pose_only_hazard_reason',
        'cancel_active_goals',
    )

    omitted_contact_checks = (
        '_fresh_retention_probe',
        '_adaptive_pressure_evidence',
        'super()._payload_hazard_reason',
        'evidence.verified',
        'minimum_force_n',
        'left_force',
        'right_force',
        'payload_pressure:',
        'payload_left_force_below_absolute_minimum',
        'payload_right_force_below_absolute_minimum',
    )
    for omitted in omitted_contact_checks:
        assert omitted not in pose_only

    # Robot collision is independent of the temporarily cleared book/finger
    # force sample epoch, so it must remain fail-closed in the pose-only path.
    assert 'target_robot_contact_latched()' in pose_only
    assert "hazard('payload_robot_contact')" in pose_only


def test_monitor_enable_happens_after_pose_watchdog_references_are_initialized():
    setup_start = SOURCE.index('        start_book = node.strict_entity_pose(BOOK)')
    loop_start = SOURCE.index(
        '        for index, target_q8 in enumerate(active_q8[1:], start=1):',
        setup_start,
    )
    setup = SOURCE[setup_start:loop_start]

    target_reference = setup.index(
        '        node.set_strict_payload_motion(\n'
        '            start_book,\n'
        "            'stationary',"
    )
    base_reference = setup.index(
        '            node._strict_payload_base_reference = start_base'
    )
    gripper_reference = setup.index(
        '            node._strict_payload_gripper_reference = float(start_gripper)'
    )
    translation_limit = setup.index(
        '            node._strict_payload_pose_probe_translation_limit_m = float('
    )
    monitor_enabled = setup.index(
        '            node._payload_monitor_enabled = True'
    )

    assert base_reference < target_reference
    assert gripper_reference < target_reference
    assert translation_limit < target_reference < monitor_enabled
