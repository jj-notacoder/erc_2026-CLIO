"""Pure gates for the post-natural-settle one-arm diagnostic."""

import math
from pathlib import Path

import numpy as np

from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (
    SHELF_FRONT_X_M,
)
from erc_phase1_solution.live_rigid_palm_natural_settle_probe import (
    EXPECTED_CAGED_APERTURE_M,
)
from erc_phase1_solution.live_rigid_palm_post_natural_settle_probe import (
    NATURAL_MAXIMUM_ARM_OUTWARD_M,
    PALM_FACE_GAP_LIMIT_M,
    PalmBearingAssessment,
    _inset_convex_polygon,
    _plan_arm_only_clearance,
    _support_mode_label,
    natural_settled_cage_guard,
    palm_bearing_support_guard,
    plan_fixed_book_palm_reposition,
    shelf_edge_proximity_geometry,
)
from erc_phase1_solution.live_rigid_palm_post_slide_transport_probe import (
    SupportAssessment,
    TransportObservation,
    _box_triangles,
    capture_attachment_anchor,
    finite_palm_support_guard,
    shelf_clearance_m,
    solve_outward_waypoints,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (
    BOOK_HALF_EXTENTS_M,
    EXPECTED_RELEASED_BASE_POSE,
    EXPECTED_RELEASED_BOOK_QUATERNION,
    BookSnapshot,
    quaternion_matrix,
)
from erc_phase1_solution.rigid_palm_preflight import (
    PALM_COLLISION_LINK,
    PreflightResult,
)


SIGNS = np.asarray(
    [
        (x, y, z)
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ],
    dtype=float,
)


def _palm_box(depth=0.080, width=0.050, thickness=0.010):
    corners = SIGNS * np.asarray(
        [0.5 * depth, 0.5 * width, 0.5 * thickness], dtype=float
    )
    return _box_triangles(corners)


def _small_book(center_z):
    return SIGNS * np.asarray([0.010, 0.010, 0.040]) + np.asarray(
        [0.0, 0.0, center_z], dtype=float
    )


def _quaternion_product(first, second):
    x1, y1, z1, w1 = np.asarray(first, dtype=float)
    x2, y2, z2, w2 = np.asarray(second, dtype=float)
    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def _settled_quaternion(angle_deg=25.0):
    half = 0.5 * math.radians(angle_deg)
    radial = np.asarray([0.0, math.sin(half), 0.0, math.cos(half)])
    result = _quaternion_product(radial, EXPECTED_RELEASED_BOOK_QUATERNION)
    return result / np.linalg.norm(result)


def _observation(*, stamp, angle_deg=25.0, shift=(0.0, 0.0, 0.0)):
    quaternion = _settled_quaternion(angle_deg)
    rotation = quaternion_matrix(quaternion)
    local = SIGNS * BOOK_HALF_EXTENTS_M
    # Match the observed natural-settle geometry: the tilted AABB can still
    # overlap the shelf front deeply even though its lower feature is caged.
    center = np.asarray([2.712, -0.146, 1.549], dtype=float)
    center += np.asarray(shift, dtype=float)
    corners = local @ rotation.T + center
    book = BookSnapshot(
        center.copy(),
        quaternion,
        np.min(corners, axis=0),
        np.max(corners, axis=0),
    )
    hand = np.eye(4)
    hand[:3, 3] = [2.64, -0.146, 1.48]
    hand[:3, 3] += np.asarray(shift, dtype=float)
    return TransportObservation(
        book=book,
        corners=corners,
        base=EXPECTED_RELEASED_BASE_POSE.copy(),
        arm=np.zeros(8),
        hand_base=hand.copy(),
        hand_world=hand,
        aperture_m=EXPECTED_CAGED_APERTURE_M,
        observed_at=float(stamp),
    )


def _safe_bearing():
    polygon = np.asarray(
        [
            [2.5, -0.3, 1.45],
            [2.8, -0.3, 1.45],
            [2.8, 0.0, 1.45],
            [2.5, 0.0, 1.45],
        ],
        dtype=float,
    )
    support = SupportAssessment(
        True,
        'ok',
        0.010,
        0.010,
        0.01075,
        0.01075,
        np.asarray([2.65, -0.15, 1.45]),
        np.asarray([0.0, 0.0, 1.0]),
        polygon,
        0.09,
    )
    return PalmBearingAssessment(
        True,
        'ok',
        support,
        0.0,
        0.0,
        0.0,
        0.099,
        0.0,
    )


def _resume_guard(reference, current, bearing=None, **changes):
    values = {
        'reference': reference,
        'current': current,
        'bearing': _safe_bearing() if bearing is None else bearing,
        'reference_time': current.observed_at + 0.01,
        'left_contact': True,
        'right_contact': True,
        'palm_contact': True,
        'unexpected_contacts': False,
    }
    values.update(changes)
    return natural_settled_cage_guard(**values)


class _CartesianFakeChain:
    """Eight-joint fake whose q[1:4] are Cartesian hand translation."""

    def __init__(self):
        self.solve_calls = 0

    def forward(self, positions):
        result = np.eye(4)
        result[:3, 3] = np.asarray(positions, dtype=float)[1:4]
        return result

    def solve(self, target, seeds, **_kwargs):
        self.solve_calls += 1
        result = np.asarray(seeds[0], dtype=float).copy()
        result[1:4] = np.asarray(target, dtype=float)[:3, 3]
        return result, 0.0

    def pose_error(self, achieved, target):
        error = np.zeros(6)
        error[:3] = (
            np.asarray(target, dtype=float)[:3, 3]
            - np.asarray(achieved, dtype=float)[:3, 3]
        )
        return error

    def link_transforms(self, positions):
        return {PALM_COLLISION_LINK: self.forward(positions)}


def test_bearing_guard_rejects_gap_that_com_projection_misses():
    palm = _palm_box()
    transform = np.eye(4)
    touching = _small_book(0.045)
    lifted = _small_book(0.045 + PALM_FACE_GAP_LIMIT_M + 0.0001)

    assert finite_palm_support_guard(
        palm, transform, lifted, (-1.0, 0.0, 0.0)
    ).safe
    accepted = palm_bearing_support_guard(
        palm, transform, touching, (-1.0, 0.0, 0.0)
    )
    rejected = palm_bearing_support_guard(
        palm, transform, lifted, (-1.0, 0.0, 0.0)
    )

    assert accepted.safe
    assert math.isclose(accepted.face_gap_m, 0.0, abs_tol=1e-12)
    assert not rejected.safe
    assert rejected.reason == 'book_not_on_selected_palm_face'


def test_outside_com_failure_retains_raw_hull_slacks_and_recovery_vector():
    corners = _small_book(0.045)
    corners[:, 0] += 0.050

    result = palm_bearing_support_guard(
        _palm_box(), np.eye(4), corners, (-1.0, 0.0, 0.0)
    )

    assert not result.safe
    assert result.reason == 'book_com_projection_outside_palm'
    assert result.raw_geometry is not None
    raw = result.raw_geometry
    assert raw.polygon_world.shape == (4, 3)
    assert raw.signed_slacks_m.shape == (4,)
    assert np.all(np.isfinite(raw.projection_world))
    assert np.all(np.isfinite(raw.signed_slacks_m))
    assert np.all(np.isfinite(raw.inset_slacks_m))
    assert math.isclose(float(np.min(raw.signed_slacks_m)), -0.010)
    assert math.isclose(float(np.min(raw.inset_slacks_m)), -0.01075)
    assert math.isclose(raw.recovery_distance_m, 0.01375)
    np.testing.assert_allclose(raw.recovery_direction_world, [1.0, 0.0, 0.0])
    assert math.isfinite(result.metrics['palm_face_signed_clearance_m'])
    assert math.isfinite(result.metrics['raw_com_projection_world_x_m'])
    assert 'raw_palm_hull_edge_3_signed_slack_m' in result.metrics
    assert all(math.isfinite(float(value)) for value in result.metrics.values())


def test_slanted_pressure_cage_retains_raw_geometry_without_proving_support():
    angle = math.radians(25.0)
    palm_world = np.eye(4)
    palm_world[:3, :3] = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=float,
    )
    result = palm_bearing_support_guard(
        _palm_box(),
        palm_world,
        _small_book(0.045),
        (-1.0, 0.0, 0.0),
    )

    assert not result.safe
    assert result.reason == 'support_surface_too_tilted'
    assert result.raw_geometry is not None
    assert all(math.isfinite(value) for value in result.raw_geometry.metrics.values())
    assert _support_mode_label(
        result,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
    ) == 'three_point_pressure_cage_without_gravity_bearing'


def test_shelf_lip_proximity_is_geometry_only_and_keeps_signed_distance():
    front = 2.755
    plane_z = 1.451858
    corners = _small_book(plane_z + 0.020)
    corners[:, 0] += front - 0.010 + 0.000015
    shelf = np.asarray(
        [
            [
                [front, -0.3, plane_z],
                [front + 0.2, -0.3, plane_z],
                [front + 0.2, 0.3, plane_z],
            ],
            [
                [front, -0.3, plane_z],
                [front + 0.2, 0.3, plane_z],
                [front, 0.3, plane_z],
            ],
        ],
        dtype=float,
    )

    result = shelf_edge_proximity_geometry(
        corners,
        shelf,
        shelf_front_x_m=front,
    )

    assert result.available
    assert result.geometry_consistent
    assert math.isclose(result.shelf_plane_z_m, plane_z)
    assert math.isclose(result.section_rear_signed_overlap_m, 0.000015)
    assert math.isclose(result.section_edge_distance_m, 0.000015)
    assert result.section_shelf_y_overlap_m > 0.0


def test_shelf_lip_proximity_accepts_official_pose_roundoff():
    front = 2.755
    plane_z = 1.451858
    corners = _small_book(plane_z + 0.020)
    corners[:, 0] += front - 0.010 + 0.000015
    # The official SDF uses 1.5708 rad, leaving about 1.1 um of z span on
    # nominally horizontal shelf triangles.
    dz = 1.102e-6
    shelf = np.asarray(
        [
            [
                [front, -0.3, plane_z - 0.5 * dz],
                [front + 0.2, -0.3, plane_z + 0.5 * dz],
                [front + 0.2, 0.3, plane_z + 0.5 * dz],
            ],
            [
                [front, -0.3, plane_z - 0.5 * dz],
                [front + 0.2, 0.3, plane_z + 0.5 * dz],
                [front, 0.3, plane_z - 0.5 * dz],
            ],
        ],
        dtype=float,
    )

    result = shelf_edge_proximity_geometry(
        corners,
        shelf,
        shelf_front_x_m=front,
    )

    assert result.available
    assert result.geometry_consistent


def test_support_inset_ignores_numerical_stl_seam_vertex():
    polygon = np.asarray(
        [
            [-0.0115, -0.0170],
            [0.0115, -0.0170],
            [0.011500004, -0.0140],
            [0.0115435, 0.0170],
            [-0.0114565, 0.0170],
        ],
        dtype=float,
    )
    # Put the middle point exactly on the long edge, as happens at a mesh
    # triangle seam after transforming the official palm STL.
    fraction = (polygon[2, 1] - polygon[1, 1]) / (
        polygon[3, 1] - polygon[1, 1]
    )
    polygon[2] = polygon[1] + fraction * (polygon[3] - polygon[1])

    result = _inset_convex_polygon(polygon, 0.00375)

    assert result.shape == (4, 2)
    assert np.all(np.ptp(result, axis=0) > 0.015)


def test_bearing_guard_rejects_excessive_palm_penetration():
    result = palm_bearing_support_guard(
        _palm_box(),
        np.eye(4),
        _small_book(0.0435),
        (-1.0, 0.0, 0.0),
    )

    assert not result.safe
    assert result.reason == 'book_palm_penetration_exceeds_limit'
    assert result.face_penetration_m > 0.001


def test_bearing_guard_rejects_disjoint_near_plane_contact_patch():
    half_angle = 0.5 * math.radians(25.0)
    rotation = quaternion_matrix(
        np.asarray([0.0, math.sin(half_angle), 0.0, math.cos(half_angle)])
    )
    corners = (
        SIGNS * np.asarray([0.080, 0.010, 0.020], dtype=float)
    ) @ rotation.T
    # Put the lowest edge exactly on the palm's z=5 mm support plane.  The
    # COM projects through the palm centre, but that edge lies beyond +x.
    corners[:, 2] += 0.005 - float(np.min(corners[:, 2]))

    projection_only = finite_palm_support_guard(
        _palm_box(), np.eye(4), corners, (-1.0, 0.0, 0.0)
    )
    result = palm_bearing_support_guard(
        _palm_box(), np.eye(4), corners, (-1.0, 0.0, 0.0)
    )

    assert projection_only.safe
    assert not result.safe
    assert result.reason == 'book_contact_patch_outside_inset_palm'
    assert not result.contact_patch_inset_overlap


def test_fixed_book_reposition_planner_selects_smallest_preflight_clear_ik():
    corners = _small_book(0.045)
    corners[:, 0] += 0.050
    start_bearing = palm_bearing_support_guard(
        _palm_box(), np.eye(4), corners, (-1.0, 0.0, 0.0)
    )
    checked: list[float] = []

    def collision_preflight(positions):
        distance = float(np.asarray(positions)[1])
        checked.append(distance)
        if distance < 0.015 - 1e-12:
            return PreflightResult(False, 'synthetic_collision', 'blocked')
        return PreflightResult(True, 'clear', 'dense fixed-book route clear')

    plan = plan_fixed_book_palm_reposition(
        chain=_CartesianFakeChain(),
        start_positions=np.zeros(8),
        base_world_transform=np.eye(4),
        book_corners_world=corners,
        palm_triangles_local=_palm_box(),
        outward_world=(-1.0, 0.0, 0.0),
        starting_bearing=start_bearing,
        collision_preflight=collision_preflight,
    )

    assert plan.safe
    assert plan.reason == 'clear'
    assert plan.candidate is not None
    assert math.isclose(plan.candidate.translation_distance_m, 0.015)
    assert math.isclose(float(plan.candidate.positions[1]), 0.015)
    assert plan.candidate.bearing.safe
    assert plan.candidate.preflight.safe
    np.testing.assert_allclose(checked, [0.014, 0.015], atol=1e-12)


def test_natural_resume_requires_stable_tilt_and_exact_three_point_cage():
    reference = _observation(stamp=1.0)
    current = _observation(stamp=1.5, shift=(0.00002, 0.0, 0.0))

    accepted = _resume_guard(reference, current)
    missing_palm = _resume_guard(reference, current, palm_contact=False)
    upright = _resume_guard(reference, _observation(stamp=1.5, angle_deg=0.0))

    assert accepted.safe
    assert 0.0 < accepted.metrics['shelf_overlap_m'] < 0.100
    assert not missing_palm.safe
    assert missing_palm.reason == 'exact_target_palm_contact_missing'
    assert not upright.safe
    assert upright.reason == 'book_not_naturally_settled'


def test_natural_resume_rejects_creep_and_unproven_bearing():
    reference = _observation(stamp=1.0)
    moved = _observation(stamp=1.5, shift=(0.00060, 0.0, 0.0))
    invalid_bearing = PalmBearingAssessment(
        False,
        'book_not_on_selected_palm_face',
        _safe_bearing().support,
        0.002,
        0.002,
        0.0,
        0.10,
        0.0,
    )

    assert _resume_guard(reference, moved).reason == 'book_not_stationary'
    assert _resume_guard(
        reference,
        _observation(stamp=1.5),
        bearing=invalid_bearing,
    ).reason == 'book_not_on_selected_palm_face'
    pressure_only = _resume_guard(
        reference,
        _observation(stamp=1.5),
        bearing=invalid_bearing,
    )
    assert pressure_only.metrics['exact_three_point_cage'] == 1.0
    assert pressure_only.metrics['gravity_bearing_support_proven'] == 0.0
    assert pressure_only.metrics['force_closure_proven'] == 0.0
    assert pressure_only.metrics[
        'pressure_cage_without_gravity_bearing'
    ] == 1.0


def test_extended_outward_solver_cap_is_bounded_before_fk():
    try:
        solve_outward_waypoints(
            None,
            np.zeros(8),
            EXPECTED_RELEASED_BASE_POSE,
            None,
            maximum_arm_outward_m=0.151,
        )
    except ValueError as exc:
        assert '5--150 mm cap' in str(exc)
    else:
        raise AssertionError('unsafe outward cap was accepted')
    assert NATURAL_MAXIMUM_ARM_OUTWARD_M == 0.130


def test_extended_solver_can_plan_natural_tilt_overlap_to_20mm_clearance():
    corners = _small_book(1.0)
    corners[:, 0] += SHELF_FRONT_X_M + 0.084 - np.max(corners[:, 0])
    book = BookSnapshot(
        np.mean(corners, axis=0),
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        np.min(corners, axis=0),
        np.max(corners, axis=0),
    )
    anchor = capture_attachment_anchor(
        book, corners, np.eye(4), observed_at=1.0
    )

    waypoints = solve_outward_waypoints(
        _CartesianFakeChain(),
        np.zeros(8),
        (0.0, 0.0, 0.0),
        anchor,
        maximum_arm_outward_m=NATURAL_MAXIMUM_ARM_OUTWARD_M,
    )

    assert len(waypoints) == 21
    assert math.isclose(waypoints[-1].outward_distance_m, 0.105)
    assert shelf_clearance_m(waypoints[-1].predicted_corners_world) >= 0.020
    assert shelf_clearance_m(waypoints[-2].predicted_corners_world) < 0.020


def test_arm_clearance_planner_is_an_explicit_no_op_when_already_clear():
    corners = _small_book(1.0)
    corners[:, 0] += SHELF_FRONT_X_M - 0.021 - np.max(corners[:, 0])
    book = BookSnapshot(
        np.mean(corners, axis=0),
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        np.min(corners, axis=0),
        np.max(corners, axis=0),
    )
    anchor = capture_attachment_anchor(
        book, corners, np.eye(4), observed_at=1.0
    )
    chain = _CartesianFakeChain()

    waypoints = _plan_arm_only_clearance(
        chain,
        np.zeros(8),
        (0.0, 0.0, 0.0),
        anchor,
        corners,
    )

    assert waypoints == ()
    assert chain.solve_calls == 0


def test_live_probe_has_read_only_default_and_no_base_or_gripper_actuator():
    source = (
        Path(__file__).resolve().parents[1]
        / 'erc_phase1_solution'
        / 'live_rigid_palm_post_natural_settle_probe.py'
    ).read_text(encoding='utf-8')

    assert "inspect_only=not arguments.confirm_arm_only_clearance" in source
    assert 'node.begin_transfer_watchdog()' in source
    assert 'node._gravity_supported_payload = False' in source
    assert source.index('node._gravity_supported_payload = False') < source.index(
        'node._gravity_supported_payload = True'
    )
    assert 'nav._accept_goal(' not in source
    assert 'nav._publish_zero(' not in source
    assert 'node._command_gripper(' not in source
    assert '_run_base_retreat(' not in source
    assert 'shelf_contact_sensor_available=False' in source
    assert 'shelf_contact_proven=False' in source
    assert 'free_gravity_carry_proven=False' in source
    executor = source[
        source.index('def _execute_arm_only_clearance('):
        source.index('\ndef _run(')
    ]
    assert executor.index('node.begin_transfer_watchdog()') < executor.index(
        'node._wait_for_retained_endpoint('
    )
    assert executor.index('node._wait_for_retained_endpoint(') < executor.index(
        'node.end_transfer_watchdog()'
    )
    assert executor.index('node._payload_hazard_reason(max_age=0.18)') < (
        executor.index('node.end_transfer_watchdog()')
    )
    assert executor.index(
        'node._post_natural_arm_motion_commanded = True'
    ) < executor.index('node._send_retained_arm_trajectory(')
    planner = source[
        source.index('def plan_fixed_book_palm_reposition('):
        source.index('\ndef _upright_reference(')
    ]
    assert '_send_' not in planner
    assert '_wait_' not in planner
    assert '_read_scene(' not in planner
