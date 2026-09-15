"""Focused static wiring checks for the one-rung live launcher."""

from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).parents[1] / 'live_physical_seed101_extract_checkpoint.py'
SOURCE = SCRIPT.read_text(encoding='utf-8')


def test_halfmillimeter_stage_has_its_own_certificate_branch_before_fallback():
    parsed = SOURCE.index("            'outward-halfmillimeter',")
    selected = SOURCE.index(
        "    halfmillimeter_outward = args.stage == 'outward-halfmillimeter'"
    )
    configured = SOURCE.index('    elif halfmillimeter_outward:')
    fallback = SOURCE.index('    else:\n        active_route = RECOVERY_ROUTE')

    assert parsed < selected < configured < fallback
    branch = SOURCE[configured:fallback]
    assert 'seed101_outward_halfmillimeter_certificate import (' in branch
    assert 'active_route = HALF_MM_ROUTE' in branch
    assert 'active_q8 = HALF_MM_ROUTE_Q8' in branch
    assert 'active_validate_certificate = validate_half_mm_certificate' in branch


def test_agreed_checkpoint_and_continuous_limits_are_bound():
    branch_start = SOURCE.index('    elif halfmillimeter_outward:')
    branch_end = SOURCE.index('    elif post_reseat_settle:', branch_start)
    branch = SOURCE[branch_start:branch_end]

    assert 'active_startup_stability_translation_limit_m = 0.00010' in branch
    assert 'active_startup_stability_rotation_limit_rad = 0.00020' in branch
    assert 'maximum_book_position_error_m' in branch
    assert 'maximum_book_rotation_error_rad' in branch
    assert 'maximum_relative_rotation_from_start_rad' in branch
    assert 'maximum_relative_yaw_from_start_rad' in branch
    assert 'maximum_cross_track_motion_m' in branch
    assert 'maximum_absolute_world_y_motion_m' in branch
    assert 'active_dense_maximum_increment_rad = 0.0005' in branch


def test_continuous_monitor_consumes_each_micro_motion_limit():
    monitor_start = SOURCE.index('        def _payload_hazard_reason(')
    monitor_end = SOURCE.index(
        '        def strict_last_payload_hazard(', monitor_start
    )
    monitor = SOURCE[monitor_start:monitor_end]

    for token in (
        '_strict_payload_progress_regression_limit_m',
        '_strict_payload_progress_overshoot_limit_m',
        '_strict_payload_cross_track_limit_m',
        '_strict_payload_lateral_y_limit_m',
        '_strict_payload_absolute_rotation_limit_rad',
        '_strict_payload_absolute_yaw_limit_rad',
        '_strict_payload_live_reference_force_n',
        '_strict_payload_live_retention_fraction',
        '_strict_payload_maximum_balance_change',
    ):
        assert token in monitor
    assert 'bilateral_pressure_retention_gate(' in monitor
    yaw_gate_start = SOURCE.index('        def strict_relative_yaw_gate(')
    yaw_gate_end = SOURCE.index(
        '        def strict_payload_observed_metrics(', yaw_gate_start
    )
    assert '_strict_payload_relative_yaw_limit_rad' in SOURCE[
        yaw_gate_start:yaw_gate_end
    ]


def test_preleg_pressure_becomes_live_floor_before_arm_dispatch():
    loop = SOURCE.index(
        '        for index, target_q8 in enumerate(active_q8[1:], start=1):'
    )
    pre_pressure = SOURCE.index(
        "pressure_evidence(f'pre_extraction_step_{index}')", loop
    )
    live_reference = SOURCE.index(
        'node._strict_payload_live_reference_force_n = (', pre_pressure
    )
    monitor_check = SOURCE.index('                require_monitor_clear()', live_reference)
    dispatch = SOURCE.index('            node.set_strict_payload_motion(', monitor_check)

    assert pre_pressure < live_reference < monitor_check < dispatch


def test_micro_geometry_is_checked_immediately_after_dwell_and_when_paused():
    loop = SOURCE.index(
        '        for index, target_q8 in enumerate(active_q8[1:], start=1):'
    )
    final_dwell = SOURCE.index(
        '        final_before_dwell = node.strict_entity_pose(BOOK)', loop
    )
    paused = SOURCE.index(
        '        paused_book = entity_pose_from_dynamic_pose(', final_dwell
    )

    assert 'progress_result = micro_outward_progress_gate(' in SOURCE[
        loop:final_dwell
    ]
    assert 'final_geometry = micro_outward_progress_gate(' in SOURCE[
        final_dwell:paused
    ]
    assert 'paused_geometry = micro_outward_progress_gate(' in SOURCE[paused:]


def test_post_final_and_paused_pressure_all_use_preleg_retention_reference():
    final_pressure = SOURCE.index(
        "        final_pressure = pressure_evidence('post_extraction_stability')"
    )
    final_retention = SOURCE.index(
        '            _require(bilateral_pressure_retention_gate(', final_pressure
    )
    pause = SOURCE.index('        pause_or_raise()', final_retention)
    paused_evidence = SOURCE.index(
        '        paused_evidence, paused_model, paused_identity = (', pause
    )
    paused_retention = SOURCE.index(
        '            _require(bilateral_pressure_retention_gate(', paused_evidence
    )

    assert final_pressure < final_retention < pause < paused_evidence < paused_retention
