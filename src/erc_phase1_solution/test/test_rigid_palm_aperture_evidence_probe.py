"""Pure tests for the diagnostic-only step-18 aperture evidence probe."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from erc_phase1_solution.live_rigid_palm_aperture_evidence_probe import (  # noqa
    APERTURE_ENDPOINT_TOLERANCE_M,
    ARM_HOLD_LIMIT_RAD,
    BASE_HOLD_LIMIT_M,
    BASE_YAW_HOLD_LIMIT_RAD,
    MINIMUM_SHELF_OVERLAP_M,
    START_APERTURE_M,
    TARGET_APERTURE_M,
    TARGET_CUMULATIVE_MOTION_LIMIT_M,
    TARGET_STEP_MOTION_LIMIT_M,
    _publish_slow_aperture,
    aperture_evidence_guard,
    dense_measured_seeded_aperture_sweep,
    measured_seeded_relative_transforms,
    reclose_authorization_guard,
    step18_aperture_resume_guard,
)
from erc_phase1_solution.live_rigid_palm_shelf_edge_cradle_probe import (  # noqa
    EXPECTED_STEP18_BOOK_MAXIMUM_X_M,
    EXPECTED_STEP18_BOOK_POSITION,
    EXPECTED_STEP18_JOINTS,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (  # noqa: E402
    BookSnapshot,
    EXPECTED_RELEASED_BASE_POSE,
    EXPECTED_RELEASED_BOOK_QUATERNION,
)
from erc_phase1_solution.rigid_palm_live_preflight import (  # noqa: E402
    MOVING_GRIPPER_LINKS,
    conservative_tool_vertex_displacement,
)
from erc_phase1_solution.rigid_palm_preflight import (  # noqa: E402
    LEFT_GRIPPER_COLLISION_LINKS,
    PALM_COLLISION_LINK,
)


def _snapshot(
    *,
    delta=(0.0, 0.0, 0.0),
    quaternion=EXPECTED_RELEASED_BOOK_QUATERNION,
    maximum_x=EXPECTED_STEP18_BOOK_MAXIMUM_X_M,
):
    offset = np.asarray(delta, dtype=float)
    position = EXPECTED_STEP18_BOOK_POSITION + offset
    half_aabb = np.asarray([0.0803, 0.0154, 0.1252])
    maximum = position + half_aabb
    maximum[0] = maximum_x + offset[0]
    return BookSnapshot(
        position=position,
        quaternion=np.asarray(quaternion, dtype=float),
        minimum=position - half_aabb,
        maximum=maximum,
    )


def _resume(**overrides):
    values = {
        'book': _snapshot(),
        'base': EXPECTED_RELEASED_BASE_POSE,
        'arm': EXPECTED_STEP18_JOINTS,
        'aperture_m': START_APERTURE_M,
        'observed_at': 10.0,
        'reference_time': 10.1,
        'left_contact': True,
        'right_contact': True,
        'palm_contact': True,
        'unexpected_contacts': False,
    }
    values.update(overrides)
    return step18_aperture_resume_guard(**values)


def test_step18_resume_accepts_only_fresh_exact_three_point_cage():
    result = _resume()

    assert result.safe
    assert result.reason == 'ok'
    assert result.metrics['shelf_overlap_m'] >= MINIMUM_SHELF_OVERLAP_M


@pytest.mark.parametrize(
    ('overrides', 'reason'),
    (
        ({'reference_time': 10.251}, 'scene_stale'),
        ({'reference_time': 9.979}, 'scene_from_future'),
        (
            {'book': _snapshot(delta=(0.00101, 0.0, 0.0))},
            'wrong_step18_book_position',
        ),
        (
            {'arm': EXPECTED_STEP18_JOINTS + 1.01 * ARM_HOLD_LIMIT_RAD},
            'wrong_step18_arm',
        ),
        (
            {'aperture_m': START_APERTURE_M + 0.000201},
            'wrong_step18_aperture',
        ),
        ({'left_contact': False}, 'left_target_contact_missing'),
        ({'right_contact': False}, 'right_target_contact_missing'),
        ({'palm_contact': False}, 'exact_target_palm_contact_missing'),
        ({'unexpected_contacts': True}, 'unexpected_contact'),
    ),
)
def test_step18_resume_rejects_wrong_or_ambiguous_state(overrides, reason):
    result = _resume(**overrides)

    assert not result.safe
    assert result.reason == reason


def _motion_guard(
    *,
    reference=None,
    previous=None,
    current=None,
    reference_base=None,
    current_base=None,
    reference_arm=None,
    current_arm=None,
    previous_aperture=START_APERTURE_M,
    aperture=TARGET_APERTURE_M,
    endpoint=True,
    left=True,
    right=True,
    palm=True,
    unexpected=False,
):
    book = _snapshot()
    base = EXPECTED_RELEASED_BASE_POSE
    arm = EXPECTED_STEP18_JOINTS
    return aperture_evidence_guard(
        book if reference is None else reference,
        book if previous is None else previous,
        book if current is None else current,
        base if reference_base is None else reference_base,
        base if current_base is None else current_base,
        arm if reference_arm is None else reference_arm,
        arm if current_arm is None else current_arm,
        previous_aperture_m=previous_aperture,
        measured_aperture_m=aperture,
        endpoint=endpoint,
        left_contact=left,
        right_contact=right,
        palm_contact=palm,
        unexpected_contacts=unexpected,
    )


def test_opening_guard_accepts_stationary_31mm_endpoint():
    result = _motion_guard()

    assert result.safe
    assert result.metrics['target_step_motion_m'] == pytest.approx(0.0)
    assert result.metrics['aperture_endpoint_error_m'] == pytest.approx(0.0)


def test_intermediate_opening_requires_palm_but_not_finger_contact():
    allowed = _motion_guard(
        aperture=0.0305,
        endpoint=False,
        left=False,
        right=False,
    )
    unsupported = _motion_guard(
        aperture=0.0305,
        endpoint=False,
        palm=False,
    )

    assert allowed.safe
    assert not unsupported.safe
    assert unsupported.reason == 'exact_target_palm_contact_missing'


@pytest.mark.parametrize(
    ('kwargs', 'reason'),
    (
        (
            {
                'current': _snapshot(
                    delta=(TARGET_STEP_MOTION_LIMIT_M + 1e-6, 0.0, 0.0)
                )
            },
            'target_step_motion',
        ),
        (
            {
                'previous': _snapshot(delta=(0.0007, 0.0, 0.0)),
                'current': _snapshot(
                    delta=(TARGET_CUMULATIVE_MOTION_LIMIT_M + 1e-6, 0.0, 0.0)
                ),
            },
            'target_cumulative_motion',
        ),
        (
            {
                'current_arm': EXPECTED_STEP18_JOINTS
                + ARM_HOLD_LIMIT_RAD
                + 1e-6
            },
            'arm_moved',
        ),
        (
            {
                'current_base': EXPECTED_RELEASED_BASE_POSE
                + np.asarray([BASE_HOLD_LIMIT_M + 1e-6, 0.0, 0.0])
            },
            'base_moved',
        ),
        (
            {
                'current_base': EXPECTED_RELEASED_BASE_POSE
                + np.asarray([0.0, 0.0, BASE_YAW_HOLD_LIMIT_RAD + 1e-6])
            },
            'base_rotated',
        ),
        (
            {
                'aperture': TARGET_APERTURE_M
                + APERTURE_ENDPOINT_TOLERANCE_M
                + 1e-6
            },
            'aperture_overshoot',
        ),
        ({'left': False}, 'left_target_contact_missing'),
        ({'right': False}, 'right_target_contact_missing'),
        ({'unexpected': True}, 'unexpected_contact'),
    ),
)
def test_opening_guard_fails_closed_at_each_motion_boundary(kwargs, reason):
    result = _motion_guard(**kwargs)

    assert not result.safe
    assert result.reason == reason


def test_reclose_requires_an_unchanged_palm_supported_pose():
    accepted = reclose_authorization_guard(
        _snapshot(),
        _snapshot(),
        EXPECTED_RELEASED_BASE_POSE,
        EXPECTED_RELEASED_BASE_POSE,
        EXPECTED_STEP18_JOINTS,
        EXPECTED_STEP18_JOINTS,
        measured_aperture_m=TARGET_APERTURE_M,
        palm_contact=True,
        unexpected_contacts=False,
    )
    moved = reclose_authorization_guard(
        _snapshot(),
        _snapshot(delta=(0.000501, 0.0, 0.0)),
        EXPECTED_RELEASED_BASE_POSE,
        EXPECTED_RELEASED_BASE_POSE,
        EXPECTED_STEP18_JOINTS,
        EXPECTED_STEP18_JOINTS,
        measured_aperture_m=TARGET_APERTURE_M,
        palm_contact=True,
        unexpected_contacts=False,
    )
    no_palm = reclose_authorization_guard(
        _snapshot(),
        _snapshot(),
        EXPECTED_RELEASED_BASE_POSE,
        EXPECTED_RELEASED_BASE_POSE,
        EXPECTED_STEP18_JOINTS,
        EXPECTED_STEP18_JOINTS,
        measured_aperture_m=TARGET_APERTURE_M,
        palm_contact=False,
        unexpected_contacts=False,
    )

    assert accepted.safe
    assert not moved.safe and moved.reason == 'target_not_unchanged'
    assert not no_palm.safe
    assert no_palm.reason == 'exact_target_palm_contact_missing'


class _FakeGripperKinematics:
    def relative_transforms(self, aperture):
        transforms = {}
        for index, link in enumerate(LEFT_GRIPPER_COLLISION_LINKS):
            transform = np.eye(4)
            if link in MOVING_GRIPPER_LINKS:
                side = -1.0 if 'left' in link.rsplit('_', 2)[-2:] else 1.0
                transform[1, 3] = side * (float(aperture) + index * 0.0001)
            transforms[link] = transform
        return transforms


def _measured_seed(kinematics):
    ideal = kinematics.relative_transforms(START_APERTURE_M)
    measured = {}
    bias = np.eye(4)
    angle = math.radians(0.4)
    bias[:3, :3] = [
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    bias[0, 3] = 0.0002
    for link in MOVING_GRIPPER_LINKS:
        measured[link] = bias @ ideal[link]
    return measured


def test_measured_seed_is_exact_and_ideal_parent_motion_reaches_endpoint():
    kinematics = _FakeGripperKinematics()
    measured = _measured_seed(kinematics)
    start = measured_seeded_relative_transforms(
        kinematics,
        measured,
        START_APERTURE_M,
        START_APERTURE_M,
    )
    endpoint = measured_seeded_relative_transforms(
        kinematics,
        measured,
        START_APERTURE_M,
        TARGET_APERTURE_M,
    )
    ideal_start = kinematics.relative_transforms(START_APERTURE_M)
    ideal_end = kinematics.relative_transforms(TARGET_APERTURE_M)

    for link in MOVING_GRIPPER_LINKS:
        np.testing.assert_allclose(start[link], measured[link], atol=1e-12)
        expected = (
            ideal_end[link]
            @ np.linalg.inv(ideal_start[link])
            @ measured[link]
        )
        np.testing.assert_allclose(endpoint[link], expected, atol=1e-12)


def test_measured_seed_rejects_any_missing_passive_link():
    kinematics = _FakeGripperKinematics()
    measured = _measured_seed(kinematics)
    measured.pop(next(iter(measured)))

    with pytest.raises(ValueError, match='all six'):
        measured_seeded_relative_transforms(
            kinematics,
            measured,
            START_APERTURE_M,
            TARGET_APERTURE_M,
        )


def test_dense_aperture_sweep_is_monotonic_and_below_vertex_step():
    kinematics = _FakeGripperKinematics()
    measured = _measured_seed(kinematics)
    meshes = tuple(
        SimpleNamespace(
            link=link,
            bounds=np.asarray([[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]]),
        )
        for link in LEFT_GRIPPER_COLLISION_LINKS
    )
    environment = SimpleNamespace(
        gripper_kinematics=kinematics,
        model=SimpleNamespace(meshes=meshes),
        config=SimpleNamespace(max_vertex_step_m=0.00020),
    )
    node = SimpleNamespace(
        chain=SimpleNamespace(
            link_transforms=lambda unused: {PALM_COLLISION_LINK: np.eye(4)}
        )
    )
    scene = SimpleNamespace(base_transform=np.eye(4))

    dense = dense_measured_seeded_aperture_sweep(
        environment=environment,
        node=node,
        scene=scene,
        arm_positions=EXPECTED_STEP18_JOINTS,
        measured_start=measured,
        start_aperture_m=START_APERTURE_M,
        target_aperture_m=TARGET_APERTURE_M,
    )

    apertures = [sample.aperture_m for sample in dense]
    assert apertures[0] == START_APERTURE_M
    assert apertures[-1] == TARGET_APERTURE_M
    assert all(
        first < second
        for first, second in zip(apertures, apertures[1:])
    )
    for first, second in zip(dense, dense[1:]):
        displacement = conservative_tool_vertex_displacement(
            environment.model,
            first.transforms,
            second.transforms,
        )
        assert displacement <= environment.config.max_vertex_step_m + 1e-12


def test_new_probe_source_has_no_arm_or_base_motion_command():
    source_path = (
        PACKAGE_ROOT
        / 'erc_phase1_solution'
        / 'live_rigid_palm_aperture_evidence_probe.py'
    )
    source = source_path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert '_move_arm_solution' not in called_attributes
    assert '_send_retained_arm_trajectory' not in called_attributes
    assert '_follow' not in called_attributes
    assert 'Gazebo entity truth' in source
    assert '--confirm-step18-aperture-evidence' in source


def test_slow_aperture_command_builds_a_valid_ros_duration():
    published = []
    node = SimpleNamespace(
        _probe_actuation_enabled=True,
        gripper_pub=SimpleNamespace(publish=published.append),
    )

    _publish_slow_aperture(node, TARGET_APERTURE_M)

    assert len(published) == 1
    point = published[0].points[0]
    assert list(point.positions) == [TARGET_APERTURE_M]
    assert point.time_from_start.sec == 2
    assert point.time_from_start.nanosec == 0
