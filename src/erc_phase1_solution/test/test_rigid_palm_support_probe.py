import math
import sys
import threading
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import erc_phase1_solution.live_rigid_palm_support_probe as palm_probe  # noqa: E402
from erc_phase1_solution.live_rigid_palm_support_probe import (  # noqa: E402
    BASE_STATIONARY_POSITION_LIMIT_M,
    CAGE_LOCK_M,
    CAGE_PRECLOSE_M,
    CAGE_PRELOAD_M,
    EXPECTED_OPEN_GRIPPER_M,
    EXPECTED_RELEASED_BASE_POSE,
    EXPECTED_RELEASED_BOOK_MAXIMUM,
    EXPECTED_RELEASED_BOOK_MINIMUM,
    EXPECTED_RELEASED_BOOK_POSITION,
    EXPECTED_RELEASED_BOOK_QUATERNION,
    EXPECTED_RELEASED_JOINTS,
    LIFT_DISTANCE_M,
    PALM_COLLISION_TOKEN,
    TARGET_BOOK_MODEL,
    V710_TOUCH,
    BookSnapshot,
    CartesianMotionMetrics,
    MotionMetrics,
    ProbeProgress,
    TangentNoise,
    _ordered_tangent_route,
    _reanchored_tangent_route,
    _preflight_cage_aperture,
    _preflight_joint_sweep,
    _probe_node_type,
    _require_audited_base_pose,
    _require_bilateral_cage,
    _require_initial_state,
    _run,
    cage_motion_is_safe,
    cage_stage_width_is_safe,
    cartesian_motion_metrics,
    learn_tangent_noise,
    measured_hand_lift_is_safe,
    motion_metrics,
    no_finger_motion_is_safe,
    quaternion_distance,
    released_state_is_expected,
    supported_release_reanchor,
    rigid_support_lift_is_safe,
    sampled_joint_segment,
    tangent_noise_is_safe,
    world_hand_pose,
)


def snapshot(
    position,
    quaternion=(0.0, 0.0, 0.0, 1.0),
    minimum=None,
    maximum=None,
):
    position = np.asarray(position, dtype=float)
    if minimum is None:
        minimum = position - np.asarray([0.08, 0.015, 0.125])
    minimum = np.asarray(minimum, dtype=float)
    if maximum is None:
        maximum = minimum + np.asarray([0.16, 0.03, 0.25])
    return BookSnapshot(
        position=position,
        quaternion=np.asarray(quaternion, dtype=float),
        minimum=minimum,
        maximum=np.asarray(maximum, dtype=float),
    )


def recorded_snapshot():
    return BookSnapshot(
        position=EXPECTED_RELEASED_BOOK_POSITION.copy(),
        quaternion=EXPECTED_RELEASED_BOOK_QUATERNION.copy(),
        minimum=EXPECTED_RELEASED_BOOK_MINIMUM.copy(),
        maximum=EXPECTED_RELEASED_BOOK_MAXIMUM.copy(),
    )


def stable_noise(reference=None):
    if reference is None:
        reference = snapshot([0.0, 0.0, 1.0])
    return TangentNoise(
        reference=reference,
        center_z_peak_to_peak_m=0.0,
        minimum_z_peak_to_peak_m=0.0,
        maximum_z_peak_to_peak_m=0.0,
        horizontal_max_deviation_m=0.0,
        rotation_max_deviation_rad=0.0,
    )


def metrics(
    *,
    translation=0.0,
    horizontal=0.0,
    rotation=0.0,
    center_rise=0.0,
    minimum_rise=0.0,
    maximum_rise=0.0,
):
    return MotionMetrics(
        translation_m=translation,
        horizontal_m=horizontal,
        rotation_rad=rotation,
        center_rise_m=center_rise,
        minimum_rise_m=minimum_rise,
        maximum_rise_m=maximum_rise,
    )


def test_package_module_import_uses_colcon_source_shape():
    module = sys.modules[BookSnapshot.__module__]
    assert module.__name__ == (
        'erc_phase1_solution.live_rigid_palm_support_probe'
    )
    assert Path(module.__file__).parent.name == 'erc_phase1_solution'


def test_sampled_joint_segment_covers_the_whole_sweep_at_bounded_steps():
    start = np.zeros(8)
    end = np.asarray([0.0, 0.051, -0.026, 0.0, 0.0, 0.0, 0.0, 0.0])

    samples = sampled_joint_segment(start, end, maximum_step=0.025)

    assert len(samples) == 4
    np.testing.assert_allclose(samples[0], start)
    np.testing.assert_allclose(samples[-1], end)
    assert all(
        np.max(np.abs(right - left)) <= 0.025
        for left, right in pairwise(samples)
    )


@pytest.mark.parametrize(
    'start,end,step',
    [
        (np.zeros(7), np.zeros(8), 0.025),
        (np.zeros(8), np.full(8, np.nan), 0.025),
        (np.zeros(8), np.zeros(8), 0.0),
    ],
)
def test_sampled_joint_segment_rejects_malformed_inputs(start, end, step):
    with pytest.raises(ValueError):
        sampled_joint_segment(start, end, maximum_step=step)


def test_full_route_is_ordered_through_the_exact_deep_tangent():
    route = _ordered_tangent_route()

    assert len(route) == 17
    assert route[0][2] == 'rigid_return_d1'
    assert route[-1][2] == 'rigid_deep_tangent'
    np.testing.assert_allclose(route[-1][0], V710_TOUCH)


def test_joint_sweep_calls_environment_preflight_and_fails_closed():
    class Node:
        def _measured_left_solution(self):
            return np.zeros(8)

        def _robot_self_collision(self, sample):
            del sample
            return None

    observed = {}

    def reject(**request):
        observed.update(request)
        return SimpleNamespace(
            safe=False,
            code='shelf_collision',
            detail='palm intersects shelf',
        )

    with pytest.raises(RuntimeError, match='shelf_collision'):
        _preflight_joint_sweep(
            Node(),
            np.full(8, 0.03),
            'approach',
            reject,
        )

    assert observed['event'] == 'approach'
    assert len(observed['joint_samples']) >= 2


def test_motion_metrics_keep_center_and_aabb_rises_separate():
    reference = snapshot(
        [2.775, -0.05, 1.575],
        minimum=[2.695, -0.065, 1.450],
        maximum=[2.855, -0.035, 1.700],
    )
    angle = math.radians(0.5)
    current = snapshot(
        [2.7753, -0.0496, 1.5752],
        [0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)],
        [2.6953, -0.0646, 1.4503],
        [2.8553, -0.0346, 1.7001],
    )

    observed = motion_metrics(reference, current)

    assert observed.horizontal_m == pytest.approx(0.0005)
    assert observed.center_rise_m == pytest.approx(0.0002)
    assert observed.minimum_rise_m == pytest.approx(0.0003)
    assert observed.maximum_rise_m == pytest.approx(0.0001)
    assert observed.rotation_rad == pytest.approx(angle)


def test_aabb_rise_without_center_rise_cannot_prove_support():
    noise = stable_noise()
    aabb_only = metrics(
        translation=0.0,
        center_rise=0.0,
        minimum_rise=LIFT_DISTANCE_M,
        maximum_rise=LIFT_DISTANCE_M,
    )

    assert not rigid_support_lift_is_safe(
        aabb_only,
        noise,
        left_contact=False,
        right_contact=False,
        unexpected_contacts=False,
        palm_contact=True,
    )


def test_quaternion_distance_is_sign_invariant_and_rejects_nonfinite_data():
    quaternion = np.asarray([0.1, -0.2, 0.3, 0.9])
    assert quaternion_distance(quaternion, -quaternion) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        quaternion_distance(
            [0.0, 0.0, np.nan, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        )


def test_initial_gate_requires_exact_recorded_joint_and_relative_book_state():
    book = recorded_snapshot()
    base = EXPECTED_RELEASED_BASE_POSE.copy()
    joints = EXPECTED_RELEASED_JOINTS.copy()

    assert released_state_is_expected(
        book,
        base,
        joints,
        EXPECTED_OPEN_GRIPPER_M,
    )

    wrong_joint = joints.copy()
    wrong_joint[4] += 0.004
    assert not released_state_is_expected(
        book,
        base,
        wrong_joint,
        EXPECTED_OPEN_GRIPPER_M,
    )

    shifted_book = recorded_snapshot()
    shifted_book.position[1] += 0.001
    shifted_book.minimum[1] += 0.001
    shifted_book.maximum[1] += 0.001
    assert not released_state_is_expected(
        shifted_book,
        base,
        joints,
        EXPECTED_OPEN_GRIPPER_M,
    )

    old_reset_target = recorded_snapshot()
    delta = 2.775 - old_reset_target.position[0]
    old_reset_target.position[0] += delta
    old_reset_target.minimum[0] += delta
    old_reset_target.maximum[0] += delta
    assert not released_state_is_expected(
        old_reset_target,
        base,
        joints,
        EXPECTED_OPEN_GRIPPER_M,
    )


def test_initial_gate_fails_on_base_or_open_gripper_mismatch():
    book = recorded_snapshot()
    base = EXPECTED_RELEASED_BASE_POSE.copy()
    base[0] += 0.0011
    assert not released_state_is_expected(
        book,
        base,
        EXPECTED_RELEASED_JOINTS,
        EXPECTED_OPEN_GRIPPER_M,
    )
    assert not released_state_is_expected(
        book,
        EXPECTED_RELEASED_BASE_POSE,
        EXPECTED_RELEASED_JOINTS,
        EXPECTED_OPEN_GRIPPER_M - 0.0011,
    )


def test_supported_release_reanchor_accepts_small_shift_but_not_large_shift():
    base = EXPECTED_RELEASED_BASE_POSE.copy()
    joints = EXPECTED_RELEASED_JOINTS.copy()
    nearby = recorded_snapshot()
    nearby.position[0] += 0.002
    nearby.minimum[0] += 0.002
    nearby.maximum[0] += 0.002

    offset = supported_release_reanchor(
        nearby, base, joints, EXPECTED_OPEN_GRIPPER_M
    )

    assert offset is not None
    expected_offset = palm_probe.relative_book_pose(base, nearby)[0] - (
        palm_probe.EXPECTED_RELEASED_BOOK_RELATIVE_POSITION
    )
    np.testing.assert_allclose(offset, expected_offset, atol=1e-12)
    far = recorded_snapshot()
    far.position[0] += 0.004
    far.minimum[0] += 0.004
    far.maximum[0] += 0.004
    assert supported_release_reanchor(
        far, base, joints, EXPECTED_OPEN_GRIPPER_M
    ) is None


def test_reanchored_route_translates_every_cartesian_endpoint():
    class LinearChain:
        @staticmethod
        def forward(joints):
            transform = np.eye(4)
            transform[:3, 3] = np.asarray(joints, dtype=float)[1:4]
            return transform

        @staticmethod
        def solve(target, seeds, **kwargs):
            del kwargs
            solved = np.asarray(seeds[-1], dtype=float).copy()
            solved[1:4] = np.asarray(target, dtype=float)[:3, 3]
            return solved, 0.0

    class Node:
        chain = LinearChain()
        joints = {'gripper_left_finger_joint': EXPECTED_OPEN_GRIPPER_M}

        @staticmethod
        def _measured_left_solution():
            return EXPECTED_RELEASED_JOINTS.copy()

    def make_pose(position, rotation):
        transform = np.eye(4)
        transform[:3, 3] = position
        transform[:3, :3] = rotation
        return transform

    nearby = recorded_snapshot()
    nearby.position[0] += 0.002
    nearby.minimum[0] += 0.002
    nearby.maximum[0] += 0.002

    route, offset = _reanchored_tangent_route(
        Node(), make_pose, nearby, EXPECTED_RELEASED_BASE_POSE
    )

    assert len(route) == len(_ordered_tangent_route())
    expected_offset = palm_probe.relative_book_pose(
        EXPECTED_RELEASED_BASE_POSE, nearby
    )[0] - palm_probe.EXPECTED_RELEASED_BOOK_RELATIVE_POSITION
    np.testing.assert_allclose(offset, expected_offset, atol=1e-12)
    for (adapted, _, event), (recorded, _, recorded_event) in zip(
        route, _ordered_tangent_route(), strict=True
    ):
        assert event == recorded_event
        np.testing.assert_allclose(
            Node.chain.forward(adapted)[:3, 3],
            Node.chain.forward(recorded)[:3, 3] + offset,
        )


def test_noise_floor_requires_multiple_samples_and_bounds_all_channels():
    samples = []
    for offset in (0.0, 0.000002, -0.000002, 0.000001, -0.000001):
        sample = snapshot(
            [1.0, 2.0, 3.0 + offset],
            minimum=[0.92, 1.985, 2.875 + offset],
            maximum=[1.08, 2.015, 3.125 + offset],
        )
        samples.append(sample)

    noise = learn_tangent_noise(samples)

    assert noise.center_z_peak_to_peak_m == pytest.approx(0.000004)
    assert tangent_noise_is_safe(noise)
    with pytest.raises(ValueError):
        learn_tangent_noise(samples[:4])

    noisy_samples = list(samples)
    noisy_samples[-1] = snapshot(
        [1.0, 2.0, 3.00005],
        minimum=[0.92, 1.985, 2.87505],
        maximum=[1.08, 2.015, 3.12505],
    )
    assert not tangent_noise_is_safe(learn_tangent_noise(noisy_samples))


def test_no_finger_gate_fails_closed_on_contact_motion_or_unexpected_contact():
    safe = metrics(translation=0.0005, horizontal=0.0004)
    assert no_finger_motion_is_safe(
        safe,
        left_contact=False,
        right_contact=False,
        unexpected_contacts=False,
    )
    assert not no_finger_motion_is_safe(
        safe,
        left_contact=True,
        right_contact=False,
        unexpected_contacts=False,
    )
    assert not no_finger_motion_is_safe(
        safe,
        left_contact=False,
        right_contact=False,
        unexpected_contacts=True,
    )
    assert not no_finger_motion_is_safe(
        metrics(translation=0.0007),
        left_contact=False,
        right_contact=False,
        unexpected_contacts=False,
    )
    assert not no_finger_motion_is_safe(
        safe,
        left_contact=False,
        right_contact=False,
        unexpected_contacts=False,
        palm_contact=True,
    )
    assert no_finger_motion_is_safe(
        safe,
        left_contact=False,
        right_contact=False,
        unexpected_contacts=False,
        palm_contact=True,
        allow_target_palm_contact=True,
    )


def test_lift_requires_center_rise_and_measured_world_hand_correlation():
    noise = stable_noise()
    book = metrics(
        translation=0.00020,
        center_rise=0.00020,
        minimum_rise=0.00020,
        maximum_rise=0.00020,
    )
    hand = CartesianMotionMetrics(LIFT_DISTANCE_M, 0.0, 0.0)

    assert rigid_support_lift_is_safe(
        book,
        noise,
        left_contact=False,
        right_contact=False,
        unexpected_contacts=False,
        palm_contact=True,
    )
    assert measured_hand_lift_is_safe(hand, book, noise)
    assert not measured_hand_lift_is_safe(
        CartesianMotionMetrics(0.0, 0.0, 0.0),
        book,
        noise,
    )
    assert not measured_hand_lift_is_safe(
        CartesianMotionMetrics(LIFT_DISTANCE_M, 0.00011, 0.0),
        book,
        noise,
    )
    assert not rigid_support_lift_is_safe(
        metrics(
            translation=0.00020,
            center_rise=0.00020,
            minimum_rise=0.00070,
            maximum_rise=0.00020,
        ),
        noise,
        left_contact=False,
        right_contact=False,
        unexpected_contacts=False,
        palm_contact=True,
    )
    assert not rigid_support_lift_is_safe(
        book,
        noise,
        left_contact=False,
        right_contact=False,
        unexpected_contacts=False,
        palm_contact=False,
    )


def test_world_hand_motion_includes_measured_base_motion():
    hand = np.eye(4)
    first = world_hand_pose([1.0, 2.0, 0.0], hand)
    second = world_hand_pose([1.00025, 2.0, 0.0], hand)

    observed = cartesian_motion_metrics(first, second)

    assert observed.horizontal_m == pytest.approx(0.00025)
    assert observed.rise_m == pytest.approx(0.0)


class FakeClock:
    def __init__(self):
        self.nanoseconds = 1_000_000_000

    def now(self):
        return SimpleNamespace(nanoseconds=self.nanoseconds)


class FakeContactBase:
    def __init__(self):
        self._lock = threading.RLock()
        self.clock = FakeClock()

    def get_clock(self):
        return self.clock

    def _on_contacts(self, message):
        del message


def contact(first, second):
    return SimpleNamespace(
        collision1=SimpleNamespace(name=first),
        collision2=SimpleNamespace(name=second),
    )


def test_transient_exact_finger_contact_stays_latched_until_cleared():
    ProbeNode = _probe_node_type(FakeContactBase, TARGET_BOOK_MODEL)
    node = ProbeNode()
    exact = contact(
        f'{TARGET_BOOK_MODEL}::book_base_link::book_collision',
        (
            'tiago_pro::gripper_left_fingertip_left_link::'
            'gripper_left_fingertip_left_link_collision'
        ),
    )

    node._on_contacts(SimpleNamespace(contacts=[exact]))
    assert node.probe_finger_latches() == (True, False)
    assert node.probe_exact_finger_sides() == (True, False)
    node._on_contacts(SimpleNamespace(contacts=[]))
    assert node.probe_finger_latches() == (True, False)
    node.clear_probe_finger_evidence()
    assert node.probe_finger_latches() == (False, False)
    assert node.probe_exact_finger_sides() == (False, False)


@pytest.mark.parametrize(
    'book_name',
    [
        'unknown::book_base_link::book_collision',
        'book_col_2_row_2_blue::book_base_link::book_collision',
    ],
)
def test_anonymous_or_wrong_book_contact_never_satisfies_target_gate(book_name):
    ProbeNode = _probe_node_type(FakeContactBase, TARGET_BOOK_MODEL)
    node = ProbeNode()
    observed = contact(
        book_name,
        (
            'tiago_pro::gripper_left_fingertip_right_link::'
            'gripper_left_fingertip_right_link_collision'
        ),
    )

    node._on_contacts(SimpleNamespace(contacts=[observed]))

    assert node.probe_finger_latches() == (False, True)
    assert node.probe_exact_finger_sides() == (False, False)
    assert node.probe_unexpected_pairs


def test_exact_target_palm_contact_is_concrete_but_not_finger_evidence():
    ProbeNode = _probe_node_type(FakeContactBase, TARGET_BOOK_MODEL)
    node = ProbeNode()
    base_contact = contact(
        f'{TARGET_BOOK_MODEL}::book_collision',
        f'tiago_pro::arm_left_7_link::{PALM_COLLISION_TOKEN}',
    )

    node._on_contacts(SimpleNamespace(contacts=[base_contact]))

    assert node.probe_finger_latches() == (False, False)
    assert node.probe_exact_finger_sides() == (False, False)
    assert node.probe_palm_latched() is True
    assert node.probe_exact_palm_contact() is True
    assert not node.probe_exact_palm_contact(since_ns=1_000_000_001)
    assert not node.probe_unexpected_pairs


def test_palm_contact_with_shelf_or_wrong_book_is_unexpected():
    ProbeNode = _probe_node_type(FakeContactBase, TARGET_BOOK_MODEL)
    node = ProbeNode()
    palm = f'tiago_pro::arm_left_7_link::{PALM_COLLISION_TOKEN}'

    node._on_contacts(
        SimpleNamespace(
            contacts=[
                contact(palm, 'erc_shelf::shelf_collision'),
                contact(
                    palm,
                    'book_col_2_row_2_blue::book_collision',
                ),
            ]
        )
    )

    assert node.probe_palm_latched() is False
    assert node.probe_exact_palm_contact() is False
    assert len(node.probe_unexpected_pairs) == 2


def test_unapproved_gripper_link_contact_with_target_is_unexpected():
    ProbeNode = _probe_node_type(FakeContactBase, TARGET_BOOK_MODEL)
    node = ProbeNode()

    node._on_contacts(
        SimpleNamespace(
            contacts=[
                contact(
                    f'{TARGET_BOOK_MODEL}::book_collision',
                    'tiago_pro::gripper_left_hand_link::hand_collision',
                )
            ]
        )
    )

    assert node.probe_finger_latches() == (False, False)
    assert node.probe_palm_latched() is False
    assert node.probe_unexpected_pairs


def test_probe_node_rejects_anonymous_target_configuration():
    with pytest.raises(ValueError):
        _probe_node_type(FakeContactBase, '')


def test_base_stationary_gate_is_tighter_than_the_support_lift():
    assert BASE_STATIONARY_POSITION_LIMIT_M < LIFT_DISTANCE_M
    reference = EXPECTED_RELEASED_BASE_POSE.copy()
    observed = reference.copy()
    _require_audited_base_pose(lambda: observed.copy(), reference=reference)
    observed[0] += BASE_STATIONARY_POSITION_LIMIT_M * 1.1
    with pytest.raises(RuntimeError, match='base moved'):
        _require_audited_base_pose(
            lambda: observed.copy(),
            reference=reference,
        )


def test_odom_pose_cannot_be_used_as_the_gazebo_world_base():
    odom_pose = np.asarray([-0.003133, 0.000003, 0.000127])

    with pytest.raises(RuntimeError, match='recorded seed-101 release pose'):
        _require_audited_base_pose(lambda: odom_pose.copy())
    assert not released_state_is_expected(
        recorded_snapshot(),
        odom_pose,
        EXPECTED_RELEASED_JOINTS,
        EXPECTED_OPEN_GRIPPER_M,
    )


def test_physical_base_pose_reads_the_tiago_gazebo_world_entity(monkeypatch):
    calls = []
    yaw = 0.4

    def fake_entity_pose(name):
        calls.append(name)
        return (
            np.asarray([2.1, -0.2, 0.03]),
            np.asarray(
                [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
            ),
        )

    monkeypatch.setattr(palm_probe, '_gazebo_entity_pose', fake_entity_pose)

    np.testing.assert_allclose(
        palm_probe._physical_base_pose(),
        [2.1, -0.2, yaw],
    )
    assert calls == ['tiago_pro']


def test_initial_state_uses_world_base_reader_and_never_odom(capsys):
    book = recorded_snapshot()
    calls = []

    class Node:
        def __init__(self):
            self.joints = {
                'gripper_left_finger_joint': EXPECTED_OPEN_GRIPPER_M
            }
            self._held_book_corners = None
            self._transport_lock_engaged = False
            self.probe_unexpected_pairs = set()

        def _measured_left_solution(self):
            return EXPECTED_RELEASED_JOINTS.copy()

        def _clear_target_contact_samples(self, **kwargs):
            del kwargs

        def clear_probe_finger_evidence(self):
            return None

        def clear_probe_palm_evidence(self):
            return None

        def _wait_sim_duration(self, duration):
            del duration
            return True

        def probe_exact_finger_sides(self, max_age):
            del max_age
            return False, False

        def probe_finger_latches(self):
            return False, False

        def probe_palm_latched(self):
            return False

    def world_base_reader():
        calls.append('world')
        return EXPECTED_RELEASED_BASE_POSE.copy()

    runtime = SimpleNamespace(
        physical_base_pose=world_base_reader,
        physical_book_bounds=bounds_reader_for(book),
    )

    _, base = _require_initial_state(Node(), runtime)

    np.testing.assert_allclose(base, EXPECTED_RELEASED_BASE_POSE)
    assert calls == ['world', 'world']
    capsys.readouterr()


def test_cage_gate_rejects_any_uncommanded_upward_center_motion():
    noise = stable_noise()
    safe = metrics(
        translation=0.0001,
        center_rise=-0.0001,
        minimum_rise=-0.0001,
        maximum_rise=-0.0001,
    )
    assert cage_motion_is_safe(safe, noise, unexpected_contacts=False)
    assert not cage_motion_is_safe(
        metrics(
            translation=0.00006,
            center_rise=0.00006,
            minimum_rise=0.00006,
            maximum_rise=0.00006,
        ),
        noise,
        unexpected_contacts=False,
    )
    assert not cage_motion_is_safe(
        safe,
        noise,
        unexpected_contacts=True,
    )


def test_cage_width_audit_requires_35_30_29_sequence_and_bounded_deltas():
    assert cage_stage_width_is_safe(
        'cage_preclose',
        CAGE_PRECLOSE_M,
        CAGE_PRECLOSE_M,
        None,
    )
    assert cage_stage_width_is_safe(
        'cage_preload',
        CAGE_PRELOAD_M,
        CAGE_PRELOAD_M,
        CAGE_PRECLOSE_M,
    )
    assert cage_stage_width_is_safe(
        'cage_transport_lock',
        CAGE_LOCK_M,
        CAGE_LOCK_M,
        CAGE_PRELOAD_M,
    )
    assert not cage_stage_width_is_safe(
        'cage_transport_lock',
        CAGE_LOCK_M,
        CAGE_LOCK_M,
        CAGE_PRECLOSE_M,
    )
    assert not cage_stage_width_is_safe(
        'cage_transport_lock',
        CAGE_PRELOAD_M,
        CAGE_PRELOAD_M,
        CAGE_PRELOAD_M,
    )


def progress_at_support():
    progress = ProbeProgress()
    for stage in ('initial', 'route', 'tangent', 'support'):
        progress.advance(stage)
    return progress


def test_progress_state_machine_cannot_report_a_false_pass():
    progress = ProbeProgress()
    with pytest.raises(RuntimeError, match='illegal probe transition'):
        progress.advance('support')
    assert progress.fail('lift failed')['passed'] is False
    with pytest.raises(RuntimeError, match='success is unavailable'):
        progress.success_payload()

    completed = progress_at_support()
    completed.advance('preload')
    completed.advance('caged')
    payload = completed.success_payload(proof='verified')
    assert payload['passed'] is True
    assert payload['next_motion_authorized'] is False


class FakeCageNode:
    def __init__(self, *, preload_contact=True, final_contact=True):
        self.gripper_preclose = CAGE_PRECLOSE_M
        self.gripper_preload = CAGE_PRELOAD_M
        self.gripper_transport_lock = CAGE_LOCK_M
        self.joints = {'gripper_left_finger_joint': 0.069}
        self.commands = []
        self._transport_lock_engaged = False
        self._exact = (False, False)
        self._preload_contact = preload_contact
        self._final_contact = final_contact
        self.probe_unexpected_pairs = set()

    def clear_probe_finger_evidence(self):
        self._exact = (False, False)

    def _command_gripper(self, position):
        self.commands.append(position)
        self.joints['gripper_left_finger_joint'] = position
        if position == CAGE_PRECLOSE_M:
            # Early bilateral evidence deliberately cannot skip preload.
            self._exact = (True, True)
        elif position == CAGE_PRELOAD_M:
            self._exact = (
                self._preload_contact,
                self._preload_contact,
            )
        elif position == CAGE_LOCK_M:
            self._exact = (True, True)
        return True

    def probe_exact_finger_sides(self, max_age=0.25):
        del max_age
        return self._exact

    def _wait_sim_duration(self, duration):
        del duration
        if self.commands and self.commands[-1] == CAGE_PRELOAD_M:
            self._exact = (self._final_contact, self._final_contact)
        return True

    def _measured_left_solution(self):
        return np.zeros(8)


def bounds_reader_for(book):
    return lambda: (
        book.position.copy(),
        book.quaternion.copy(),
        book.minimum.copy(),
        book.maximum.copy(),
    )


def test_fake_cage_requires_preload_even_when_preclose_has_contact(capsys):
    lifted = snapshot([0.0, 0.0, 1.0])
    node = FakeCageNode()
    progress = progress_at_support()

    returned = _require_bilateral_cage(
        node,
        bounds_reader_for(lifted),
        lifted,
        stable_noise(lifted),
        progress,
    )

    assert node.commands == [CAGE_PRECLOSE_M, CAGE_PRELOAD_M]
    assert progress.stage == 'caged'
    assert node._transport_lock_engaged is False
    np.testing.assert_allclose(returned.position, lifted.position)
    capsys.readouterr()


def test_fake_cage_failure_never_commands_lock_or_allows_pass(capsys):
    lifted = snapshot([0.0, 0.0, 1.0])
    node = FakeCageNode(preload_contact=False)
    progress = progress_at_support()

    with pytest.raises(RuntimeError, match='preload did not acquire'):
        _require_bilateral_cage(
            node,
            bounds_reader_for(lifted),
            lifted,
            stable_noise(lifted),
            progress,
        )

    assert node.commands == [CAGE_PRECLOSE_M, CAGE_PRELOAD_M]
    assert progress.stage == 'support'
    assert node._transport_lock_engaged is False
    with pytest.raises(RuntimeError, match='success is unavailable'):
        progress.success_payload()
    capsys.readouterr()


def test_fake_cage_requires_fresh_bilateral_contact_after_preload(capsys):
    lifted = snapshot([0.0, 0.0, 1.0])
    node = FakeCageNode(final_contact=False)
    progress = progress_at_support()

    with pytest.raises(RuntimeError, match='final fresh bilateral'):
        _require_bilateral_cage(
            node,
            bounds_reader_for(lifted),
            lifted,
            stable_noise(lifted),
            progress,
        )

    assert node.commands == [CAGE_PRECLOSE_M, CAGE_PRELOAD_M]
    assert progress.stage == 'preload'
    assert node._transport_lock_engaged is False
    capsys.readouterr()


def test_cage_environment_preflight_runs_before_each_command(capsys):
    lifted = snapshot([0.0, 0.0, 1.0])
    node = FakeCageNode()
    progress = progress_at_support()

    class Preflight:
        def __init__(self):
            self.requests = []

        def preflight_aperture_sweep(self, **request):
            commands_before_event = {
                'cage_preclose': [],
                'cage_preload': [CAGE_PRECLOSE_M],
            }
            assert node.commands == commands_before_event[request['event']]
            self.requests.append(request)
            return SimpleNamespace(safe=True, code='clear', detail='clear')

    preflight = Preflight()
    _require_bilateral_cage(
        node,
        bounds_reader_for(lifted),
        lifted,
        stable_noise(lifted),
        progress,
        preflight,
    )

    assert [request['event'] for request in preflight.requests] == [
        'cage_preclose',
        'cage_preload',
    ]
    capsys.readouterr()


def test_rejected_cage_preflight_prevents_the_controller_command():
    node = FakeCageNode()

    class Rejected:
        def preflight_aperture_sweep(self, **request):
            del request
            return SimpleNamespace(
                safe=False,
                code='tool_shelf_collision',
                detail='finger intersects shelf',
            )

    with pytest.raises(RuntimeError, match='tool_shelf_collision'):
        _preflight_cage_aperture(
            node,
            Rejected(),
            'cage_preclose',
            CAGE_PRECLOSE_M,
        )
    assert node.commands == []


def test_preflight_only_virtual_cage_translates_the_lifted_target(
    monkeypatch,
    capsys,
):
    start = EXPECTED_RELEASED_JOINTS.copy()
    route_target = start + 0.001
    lift_target = route_target + 0.0001
    aperture_requests = []

    class Node:
        def _measured_left_solution(self):
            return start.copy()

    class Preflight:
        def preflight_aperture_sweep(self, **request):
            aperture_requests.append(request)
            return SimpleNamespace(safe=True, code='clear', detail='clear')

    monkeypatch.setattr(
        palm_probe,
        '_reanchored_tangent_route',
        lambda *args: (
            [(route_target, 1.0, 'route_sample')],
            np.zeros(3),
        ),
    )
    monkeypatch.setattr(
        palm_probe,
        '_preflight_joint_sweep',
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        palm_probe,
        '_solve_exact_lift',
        lambda *args, **kwargs: lift_target.copy(),
    )
    runtime = SimpleNamespace(
        environment_preflight=Preflight(),
        pose_matrix=object(),
        physical_base_pose=lambda: EXPECTED_RELEASED_BASE_POSE.copy(),
        physical_book_bounds=lambda: (
            EXPECTED_RELEASED_BOOK_POSITION.copy(),
            EXPECTED_RELEASED_BOOK_QUATERNION.copy(),
            EXPECTED_RELEASED_BOOK_MINIMUM.copy(),
            EXPECTED_RELEASED_BOOK_MAXIMUM.copy(),
        ),
    )

    palm_probe._run_preflight_only(
        Node(),
        runtime,
        EXPECTED_RELEASED_BASE_POSE.copy(),
    )

    assert [request['event'] for request in aperture_requests] == [
        'cage_preclose',
        'cage_preload',
    ]
    for request in aperture_requests:
        np.testing.assert_allclose(request['joint_positions'], lift_target)
        assert request['target_book_translation'] == (
            0.0,
            0.0,
            LIFT_DISTANCE_M,
        )
    capsys.readouterr()


def test_preflight_only_guards_every_manipulation_dispatch_path():
    calls = []

    class Base:
        def __init__(self):
            # Mirrors Node.create_subscription capturing the bound callback.
            self.command_callback = self._on_command

        def _publish_status(self, event, **fields):
            calls.append(('status', event, fields))

        def _on_command(self, message):
            calls.append(('command', message.data))
            self._command_gripper(0.01)

        def _run_command(self, command):
            calls.append(('run_command', command))

        def _follow(self, *args, **kwargs):
            calls.append(('follow', args, kwargs))
            return True

        def _cancel_goal_and_confirm(self, *args, **kwargs):
            calls.append(('cancel', args, kwargs))
            return True

        def _send_retained_arm_trajectory(self, *args, **kwargs):
            calls.append(('retained', args, kwargs))
            return True, False

        def _command_gripper(self, position):
            calls.append(('gripper', position))
            return True

    Guarded = _probe_node_type(
        Base,
        TARGET_BOOK_MODEL,
        preflight_only=True,
    )
    guarded = Guarded()

    guarded.command_callback(SimpleNamespace(data='pick'))

    assert calls == [
        (
            'status',
            'rejected',
            {'command': 'pick', 'reason': 'preflight_only'},
        )
    ]
    assert guarded._probe_actuation_enabled is False
    guarded._probe_actuation_enabled = True
    guarded.command_callback(SimpleNamespace(data='cancel'))
    assert calls[-1] == (
        'status',
        'rejected',
        {'command': 'cancel', 'reason': 'preflight_only'},
    )
    guarded._run_command('stow')
    assert calls[-1] == (
        'status',
        'rejected',
        {'command': 'stow', 'reason': 'preflight_only'},
    )
    with pytest.raises(RuntimeError, match='controller action goals'):
        guarded._follow(object(), ['joint'], [0.0], 1.0)
    with pytest.raises(RuntimeError, match='controller cancellation'):
        guarded._cancel_goal_and_confirm(object(), object())
    with pytest.raises(RuntimeError, match='retained arm goals'):
        guarded._send_retained_arm_trajectory(object(), 1.0, [], 'pick')
    with pytest.raises(RuntimeError, match='gripper commands'):
        guarded._command_gripper(0.01)
    assert len(calls) == 3

    Live = _probe_node_type(Base, TARGET_BOOK_MODEL)
    live = Live()
    live.command_callback(SimpleNamespace(data='open_gripper'))
    live._run_command('stow')
    assert live._probe_actuation_enabled is True
    assert ('command', 'open_gripper') in calls
    assert ('run_command', 'stow') in calls
    assert ('gripper', 0.01) in calls
    assert live._follow(object(), ['joint'], [0.0], 1.0) is True
    assert live._cancel_goal_and_confirm(object(), object()) is True
    assert live._send_retained_arm_trajectory(
        object(),
        1.0,
        [],
        'pick',
    ) == (True, False)
    assert live._command_gripper(0.02) is True


def test_preflight_only_never_constructs_or_commands_navigation(monkeypatch):
    events = []

    class Base:
        def __init__(self):
            self.joints = {str(index): 0.0 for index in range(8)}
            self._lock = threading.RLock()

        def _publish_status(self, event, **fields):
            events.append(('status', event, fields))

        def _on_command(self, message):
            events.append(('unguarded_command', message.data))

        def _command_gripper(self, position):
            events.append(('gripper_command', position))
            return True

        def destroy_node(self):
            events.append('destroy_node')

    class Executor:
        def __init__(self, num_threads):
            assert num_threads == 8

        def add_node(self, node):
            events.append(('add_node', type(node).__name__))
            events.append(
                ('actuation_enabled', node._probe_actuation_enabled)
            )
            node._on_command(SimpleNamespace(data='pick'))

        def spin(self):
            return None

        def shutdown(self):
            events.append('shutdown')

    def forbidden_navigation():
        raise AssertionError('preflight-only must not construct NavigationNode')

    monkeypatch.setattr(
        palm_probe,
        '_require_initial_state',
        lambda node, runtime: (
            recorded_snapshot(),
            EXPECTED_RELEASED_BASE_POSE.copy(),
        ),
    )
    monkeypatch.setattr(
        palm_probe,
        '_run_preflight_only',
        lambda node, runtime, base: events.append('preflight_only'),
    )
    runtime = SimpleNamespace(
        BOOK=TARGET_BOOK_MODEL,
        ManipulationNode=Base,
        NavigationNode=forbidden_navigation,
        MultiThreadedExecutor=Executor,
        environment_preflight=object(),
    )

    _run(runtime, preflight_only=True)

    assert 'preflight_only' in events
    assert 'shutdown' in events
    assert 'destroy_node' in events
    assert ('actuation_enabled', False) in events
    assert (
        'status',
        'rejected',
        {'command': 'pick', 'reason': 'preflight_only'},
    ) in events
    tuple_events = [event for event in events if isinstance(event, tuple)]
    assert not any(event[0] == 'unguarded_command' for event in tuple_events)
    assert not any(event[0] == 'gripper_command' for event in tuple_events)


def test_executor_spin_is_joined_before_node_destruction(monkeypatch):
    events = []

    class Base:
        def __init__(self):
            self.joints = {str(index): 0.0 for index in range(8)}
            self._lock = threading.RLock()

        def destroy_node(self):
            events.append('destroy_node')

    class Executor:
        def __init__(self, num_threads):
            assert num_threads == 8

        def add_node(self, node):
            del node

        def spin(self):
            events.append('spin')
            raise RuntimeError('synthetic spin failure')

        def shutdown(self):
            events.append('shutdown')

    class SpinThread:
        def __init__(self, *, target, name, daemon):
            assert name == 'rigid-palm-probe-executor'
            assert daemon is True
            self.target = target

        def start(self):
            events.append('thread_start')
            self.target()

        def join(self):
            events.append('thread_join')

    def forbidden_navigation():
        raise AssertionError('preflight-only must not construct NavigationNode')

    monkeypatch.setattr(
        palm_probe,
        'threading',
        SimpleNamespace(Thread=SpinThread),
    )
    monkeypatch.setattr(
        palm_probe,
        '_require_initial_state',
        lambda node, runtime: (
            recorded_snapshot(),
            EXPECTED_RELEASED_BASE_POSE.copy(),
        ),
    )
    monkeypatch.setattr(
        palm_probe,
        '_run_preflight_only',
        lambda node, runtime, base: events.append('preflight_only'),
    )
    runtime = SimpleNamespace(
        BOOK=TARGET_BOOK_MODEL,
        ManipulationNode=Base,
        NavigationNode=forbidden_navigation,
        MultiThreadedExecutor=Executor,
        environment_preflight=object(),
    )

    with pytest.raises(RuntimeError, match='executor spin failed'):
        _run(runtime, preflight_only=True)

    assert events.index('shutdown') < events.index('thread_join')
    assert events.index('thread_join') < events.index('destroy_node')
