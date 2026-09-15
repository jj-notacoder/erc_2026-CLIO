"""Focused no-ROS checks for the strict relative-yaw debounce."""

from __future__ import annotations

import ast
import copy
import importlib.util
import math
from pathlib import Path
import sys
import threading
import time
from typing import Mapping, Optional, Tuple

import pytest


SCRIPT = Path(__file__).parents[1] / (
    'live_physical_seed101_extract_checkpoint.py'
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location('seed101_yaw_gate', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)

from erc_phase1_solution import (  # noqa: E402
    seed101_outward_halfmillimeter_midroute_settle_certificate as midroute,
)


SOURCE = SCRIPT.read_text(encoding='utf-8')


def nested_method(name: str):
    """Compile one nested class method without entering launcher ``main``."""

    tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
    strict_held_classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == 'StrictHeldNode'
    ]
    assert len(strict_held_classes) == 1
    matches = [
        node
        for node in strict_held_classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(matches) == 1
    module = ast.Module(body=[copy.deepcopy(matches[0])], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        'Mapping': Mapping,
        'Optional': Optional,
        'Tuple': Tuple,
        'EntityPose': probe.EntityPose,
        'PAYLOAD_TARGET_POSE_MAXIMUM_WALL_AGE_S': (
            probe.PAYLOAD_TARGET_POSE_MAXIMUM_WALL_AGE_S
        ),
        'consecutive_soft_limit_gate': probe.consecutive_soft_limit_gate,
        'math': math,
        'time': time,
    }
    exec(compile(module, str(SCRIPT), 'exec'), namespace)
    return namespace[name]


STRICT_RELATIVE_YAW_GATE = nested_method('strict_relative_yaw_gate')
STRICT_ENTITY_POSE_WITH_GENERATION = nested_method(
    'strict_entity_pose_with_generation'
)


class YawGateState:
    def __init__(self) -> None:
        self._strict_hazard_lock = threading.Lock()
        self._strict_payload_relative_yaw_limit_rad = 0.00045
        self._strict_payload_relative_yaw_hard_limit_rad = 0.00080
        self._strict_payload_relative_yaw_consecutive_samples_required = 2
        self._strict_payload_relative_yaw_consecutive_samples = 0
        self._strict_payload_relative_yaw_last_generation = -1
        self._strict_payload_max_relative_yaw_rad = 0.0
        self._strict_payload_max_consecutive_soft_yaw_samples = 0

    strict_relative_yaw_gate = STRICT_RELATIVE_YAW_GATE


def test_same_pose_generation_is_not_double_counted():
    state = YawGateState()

    assert state.strict_relative_yaw_gate(0.00046, 10) is None
    assert state._strict_payload_relative_yaw_consecutive_samples == 1
    assert state.strict_relative_yaw_gate(0.00047, 10) is None
    assert state._strict_payload_relative_yaw_consecutive_samples == 1
    assert state._strict_payload_relative_yaw_last_generation == 10


def test_at_or_below_soft_limit_resets_the_consecutive_count():
    state = YawGateState()

    assert state.strict_relative_yaw_gate(0.00046, 1) is None
    assert state._strict_payload_relative_yaw_consecutive_samples == 1
    assert state.strict_relative_yaw_gate(0.00045, 2) is None
    assert state._strict_payload_relative_yaw_consecutive_samples == 0
    assert state.strict_relative_yaw_gate(0.00046, 3) is None
    assert state._strict_payload_relative_yaw_consecutive_samples == 1


def test_two_distinct_over_soft_generations_trip():
    state = YawGateState()

    assert state.strict_relative_yaw_gate(0.00046, 20) is None
    fault = state.strict_relative_yaw_gate(0.00047, 21)

    assert fault == {
        'kind': 'debounced_limit_sustained_trip',
        'yaw_error_rad': 0.00047,
        'yaw_limit_rad': 0.00045,
        'consecutive_samples': 2,
        'consecutive_samples_required': 2,
    }


def test_hard_limit_is_inclusive_but_any_excess_trips_immediately():
    state = YawGateState()

    assert state.strict_relative_yaw_gate(0.00080, 30) is None
    state = YawGateState()
    fault = state.strict_relative_yaw_gate(0.000800001, 30)

    assert fault == {
        'kind': 'debounced_limit_hard_trip',
        'yaw_error_rad': 0.000800001,
        'yaw_limit_rad': 0.00080,
        'consecutive_samples': 2,
        'consecutive_samples_required': 2,
    }


def test_both_live_pose_watchdogs_delegate_to_generation_aware_gate():
    start = SOURCE.index('        def _payload_hazard_reason(')
    middle = SOURCE.index('        def strict_pose_only_hazard_reason(', start)
    end = SOURCE.index('        def cancel_active_goals(', middle)

    for method_source in (SOURCE[start:middle], SOURCE[middle:end]):
        assert 'yaw_fault = self.strict_relative_yaw_gate(' in method_source
        assert 'target_generation,' in method_source
        assert 'self.strict_entity_pose_with_generation(BOOK)' in method_source
        assert 'self.strict_entity_generation(BOOK)' not in method_source


def test_pose_and_generation_are_read_under_one_entity_lock():
    first_pose = probe.EntityPose(
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    second_pose = probe.EntityPose(
        (4.0, 5.0, 6.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    pose_read = threading.Event()
    update_attempted = threading.Event()

    class BlockingPoseMap(dict):
        def get(self, key, default=None):
            pose_read.set()
            assert update_attempted.wait(timeout=1.0)
            return super().get(key, default)

    class EntityState:
        strict_entity_pose_with_generation = STRICT_ENTITY_POSE_WITH_GENERATION

        def __init__(self) -> None:
            self._strict_entity_lock = threading.Lock()
            self._strict_entity_poses = BlockingPoseMap(
                {probe.BOOK: first_pose}
            )
            self._strict_entity_wall_ns = {
                probe.BOOK: time.monotonic_ns(),
            }
            self._strict_entity_generations = {probe.BOOK: 11}

    state = EntityState()

    def update_pose() -> None:
        assert pose_read.wait(timeout=1.0)
        update_attempted.set()
        with state._strict_entity_lock:
            state._strict_entity_poses[probe.BOOK] = second_pose
            state._strict_entity_wall_ns[probe.BOOK] = time.monotonic_ns()
            state._strict_entity_generations[probe.BOOK] = 12

    updater = threading.Thread(target=update_pose)
    updater.start()
    observed_pose, observed_generation = (
        state.strict_entity_pose_with_generation(probe.BOOK)
    )
    updater.join(timeout=1.0)

    assert not updater.is_alive()
    assert observed_pose == first_pose
    assert observed_generation == 11
    assert state._strict_entity_poses[probe.BOOK] == second_pose
    assert state._strict_entity_generations[probe.BOOK] == 12


def test_midroute_settle_binds_zero_command_route_and_yaw_debounce_policy():
    valid, metrics = midroute.validate_certificate()

    assert valid
    assert midroute.STAGE == 'outward-halfmillimeter-midroute-settle'
    assert midroute.ROUTE_Q8 == (midroute.CHECKPOINT_LEFT_Q8,)
    assert midroute.COMMAND_Q8 == ()
    assert metrics['route_rows'] == 1
    assert metrics['command_rows'] == 0

    parsed = SOURCE.index("            'outward-halfmillimeter-midroute-settle',")
    selected = SOURCE.index(
        '    halfmillimeter_midroute_settle = (\n'
        "        args.stage == 'outward-halfmillimeter-midroute-settle'"
    )
    configured = SOURCE.index('    elif halfmillimeter_midroute_settle:')
    branch_end = SOURCE.index('    elif post_reseat_settle:', configured)
    assert parsed < selected < configured < branch_end
    branch = SOURCE[configured:branch_end]

    assert (
        'seed101_outward_halfmillimeter_midroute_settle_certificate import ('
        in branch
    )
    assert 'active_route = MIDROUTE_ROUTE' in branch
    assert 'active_q8 = MIDROUTE_ROUTE_Q8' in branch
    assert 'active_validate_certificate = validate_midroute_certificate' in branch
    assert 'active_peel_delta = (0.0, 0.0, 0.0)' in branch
    assert 'active_peel_duration = 0.0' in branch
    assert (
        "MIDROUTE_YAW_POLICY['relative_yaw_soft_limit_rad']"
        in branch
    )
    assert (
        "MIDROUTE_YAW_POLICY['relative_yaw_immediate_hard_limit_rad']"
        in branch
    )
    assert "MIDROUTE_YAW_POLICY['consecutive_samples_required']" in branch
    assert (
        'for index, target_q8 in enumerate(active_q8[1:], start=1):'
        in SOURCE
    )
    assert 'route_steps=len(active_q8) - 1' in SOURCE
    assert 'left_arm_trajectory_commanded=bool(len(active_q8) > 1)' in SOURCE


def midroute_supported_book_gate(observed, stable_reference):
    policy = midroute.FINAL_DWELL_POLICY
    return probe.supported_book_state_gate(
        observed,
        stable_reference,
        support_floor_world_z_m=midroute.SUPPORT_FLOOR_WORLD_Z_M,
        maximum_rotation_rad=float(
            policy['maximum_absolute_rotation_to_stable_reference_rad']
        ),
        maximum_yaw_component_rad=float(
            policy['maximum_absolute_yaw_to_stable_reference_rad']
        ),
        minimum_floor_signed_distance_m=float(policy['floor_signed_min_m']),
        maximum_floor_signed_distance_m=float(policy['floor_signed_max_m']),
    )


def test_supported_book_gate_accepts_the_bound_stable_supported_pose():
    reference = probe.EntityPose(
        midroute.STABLE_REFERENCE_BOOK_POSITION_WORLD_M,
        midroute.STABLE_REFERENCE_BOOK_QUATERNION_XYZW,
    )

    result = midroute_supported_book_gate(reference, reference)

    assert result.ok
    assert result.reason == 'supported_book_state_verified'


@pytest.mark.parametrize(
    ('axis_quaternion', 'reason'),
    (
        (
            (math.sin(0.001 / 2.0), 0.0, 0.0, math.cos(0.001 / 2.0)),
            'checkpoint_absolute_rotation_exceeded_limit',
        ),
        (
            (0.0, 0.0, math.sin(0.0007 / 2.0), math.cos(0.0007 / 2.0)),
            'checkpoint_absolute_yaw_exceeded_limit',
        ),
    ),
)
def test_supported_book_gate_rejects_orientation_limits(axis_quaternion, reason):
    reference = probe.EntityPose((0.0, 0.0, 1.6), (0.0, 0.0, 0.0, 1.0))
    observed = probe.EntityPose(reference.position, axis_quaternion)

    result = midroute_supported_book_gate(observed, reference)

    assert not result.ok
    assert result.reason == reason


@pytest.mark.parametrize(
    ('z_offset_m', 'reason'),
    (
        (-0.0002, 'supported_book_shelf_penetration_exceeded_limit'),
        (0.0003, 'supported_book_lost_shelf_support'),
    ),
)
def test_supported_book_gate_rejects_floor_limits(z_offset_m, reason):
    reference = probe.EntityPose(
        midroute.STABLE_REFERENCE_BOOK_POSITION_WORLD_M,
        midroute.STABLE_REFERENCE_BOOK_QUATERNION_XYZW,
    )
    observed = probe.EntityPose(
        (
            reference.position[0],
            reference.position[1],
            reference.position[2] + z_offset_m,
        ),
        reference.quaternion,
    )

    result = midroute_supported_book_gate(observed, reference)

    assert not result.ok
    assert result.reason == reason
