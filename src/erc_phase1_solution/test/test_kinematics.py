"""Pure NumPy/URDF checks for the official v1.0.3 left-arm interface."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from erc_phase1_solution.kinematics import (
    URDFChain,
    interpolate_joint_waypoints,
    load_urdf_collision_meshes,
    oriented_box_intersects_triangles,
    pose_matrix,
    triangle_meshes_intersect,
    triangles_intersect,
)
from erc_phase1_solution.motion_profiles import (
    CARRY,
    HOME,
    IK_JOINTS,
    OFFER,
    PREGRASP,
    RIGHT_ARM_JOINTS,
    RIGHT_HOME,
    SUPPORTED_CARRY,
    bin_place_orientations,
    shelf_grasp_orientations,
    shelf_pinch_orientations,
    supported_bin_place_orientations,
)


@pytest.fixture(scope='module')
def official_chain() -> URDFChain:
    urdf = (
        Path(__file__).resolve().parents[2]
        / 'erc_description'
        / 'urdf'
        / 'tiago_pro.urdf'
    )
    if not urdf.exists():
        pytest.skip('official erc_description URDF is not in this workspace')
    return URDFChain.from_urdf(
        urdf,
        'base_footprint',
        'gripper_left_grasping_link',
        IK_JOINTS,
    )


@pytest.fixture(scope='module')
def official_right_chain() -> URDFChain:
    urdf = (
        Path(__file__).resolve().parents[2]
        / 'erc_description'
        / 'urdf'
        / 'tiago_pro.urdf'
    )
    if not urdf.exists():
        pytest.skip('official erc_description URDF is not in this workspace')
    return URDFChain.from_urdf(
        urdf,
        'base_footprint',
        'gripper_right_grasping_link',
        ('torso_lift_joint', *RIGHT_ARM_JOINTS),
    )


def _collision_meshes(links):
    source_root = Path(__file__).resolve().parents[2]
    urdf = source_root / 'erc_description' / 'urdf' / 'tiago_pro.urdf'
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
    return load_urdf_collision_meshes(
        urdf,
        links,
        package_paths.__getitem__,
    )


def test_official_head_and_camera_collision_geometry_loads():
    links = ('head_1_link', 'head_2_link', 'head_front_camera_link')
    meshes = _collision_meshes(links)

    assert {mesh.link for mesh in meshes} == set(links)
    assert all(len(mesh.triangles) > 0 for mesh in meshes)


def test_motion_profiles_fit_official_joint_limits(official_chain):
    for profile in (HOME, OFFER, PREGRASP, CARRY, SUPPORTED_CARRY):
        assert profile.shape == official_chain.lower.shape
        assert np.all(profile >= official_chain.lower)
        assert np.all(profile <= official_chain.upper)


def test_official_chain_uses_urdf_hard_limits():
    source_root = Path(__file__).resolve().parents[2]
    chain = URDFChain.from_urdf(
        source_root / 'erc_description' / 'urdf' / 'tiago_pro.urdf',
        'base_footprint',
        'gripper_left_grasping_link',
        IK_JOINTS,
    )

    assert chain.lower[0] == pytest.approx(-0.001)
    assert chain.upper[0] == pytest.approx(0.35)
    assert chain.lower[4] == pytest.approx(-2.443460952792061)
    assert chain.upper[4] == pytest.approx(1.1344640137963142)


def test_unused_right_arm_uses_official_mirrored_home_posture():
    assert RIGHT_ARM_JOINTS == tuple(
        f'arm_right_{index}_joint' for index in range(1, 8)
    )
    assert RIGHT_HOME == pytest.approx(
        [-0.36, -1.83, -0.47, -2.35, 0.0, -1.20, 0.0]
    )


@pytest.mark.parametrize('height', [0.605, 0.935, 1.265, 1.595])
def test_pinch_orientations_keep_finger_axis_horizontal(height):
    for rotation in shelf_pinch_orientations(height):
        assert abs(rotation[2, 1]) <= 1e-12
        assert abs(rotation[1, 1]) >= math.cos(0.50) - 1e-12


def test_top_row_orientation_has_no_shelf_normal_finger_stagger():
    rotations = shelf_pinch_orientations(1.583)

    assert len(rotations) == 1
    jaw_axis = rotations[0][:, 1]
    assert jaw_axis[0] == pytest.approx(0.0, abs=1e-12)
    assert jaw_axis[1] == pytest.approx(-1.0)
    assert jaw_axis[2] == pytest.approx(0.0, abs=1e-12)
    assert rotations[0][0, 0] == pytest.approx(math.cos(0.50))


def test_bin_orientations_keep_finger_axis_horizontal():
    rotations = bin_place_orientations()
    assert rotations[0] == pytest.approx(
        np.diag([1.0, -1.0, -1.0]),
        abs=1e-12,
    )
    for rotation in rotations:
        assert abs(rotation[2, 1]) <= 1e-12
        assert abs(rotation[1, 1]) == pytest.approx(1.0)


def test_supported_bin_orientations_keep_finger_axis_vertical():
    rotations = supported_bin_place_orientations()

    assert len(rotations) == 3
    assert rotations[0][2, 1] > 0.96
    for rotation in rotations:
        assert abs(rotation[2, 1]) >= math.cos(0.50) - 1e-12


def test_carry_profile_is_tucked_inside_previous_offer_reach(official_chain):
    carry_x = official_chain.forward(CARRY)[0, 3]
    offer_x = official_chain.forward(OFFER)[0, 3]
    assert carry_x < 0.50
    assert offer_x > 0.80


def test_link_positions_end_at_forward_tip(official_chain):
    points = official_chain.link_positions(HOME)
    assert points.shape == (len(official_chain.joints), 3)
    assert points[-1] == pytest.approx(official_chain.forward(HOME)[:3, 3])


def test_link_transforms_end_at_forward_tip(official_chain):
    transforms = official_chain.link_transforms(HOME)

    assert set(transforms) == {joint.child for joint in official_chain.joints}
    assert transforms['gripper_left_grasping_link'] == pytest.approx(
        official_chain.forward(HOME)
    )


def test_oriented_box_triangle_intersection_uses_full_sat():
    corners = np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    )
    crossing = np.asarray([[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    clear = crossing + np.asarray([0.0, 0.0, 2.1])

    assert oriented_box_intersects_triangles(corners, crossing)
    assert not oriented_box_intersects_triangles(corners, clear)


def test_oriented_box_detects_containment_inside_closed_mesh():
    inner_corners = np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    )
    outer_vertices = np.asarray(
        [
            (x, y, z)
            for x in (-3.0, 3.0)
            for y in (-3.0, 3.0)
            for z in (-3.0, 3.0)
        ]
    )
    faces = np.asarray(
        [
            (0, 1, 3), (0, 3, 2),
            (4, 6, 7), (4, 7, 5),
            (0, 4, 5), (0, 5, 1),
            (2, 3, 7), (2, 7, 6),
            (0, 2, 6), (0, 6, 4),
            (1, 5, 7), (1, 7, 3),
        ]
    )

    assert not oriented_box_intersects_triangles(
        inner_corners,
        outer_vertices[faces],
    )
    assert oriented_box_intersects_triangles(
        inner_corners,
        outer_vertices[faces],
        closed_surface=True,
    )


def test_triangle_intersection_handles_crossing_and_coplanar_facets():
    horizontal = np.asarray(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    crossing = np.asarray(
        [[0.0, -0.5, -1.0], [0.0, -0.5, 1.0], [0.0, 0.5, 0.0]]
    )
    coplanar_overlap = horizontal + np.asarray([0.25, 0.0, 0.0])
    coplanar_clear = horizontal + np.asarray([3.0, 0.0, 0.0])

    assert triangles_intersect(horizontal, crossing)
    assert triangles_intersect(horizontal, coplanar_overlap)
    assert not triangles_intersect(horizontal, coplanar_clear)


def test_triangle_intersection_preserves_exact_touching_tolerance_boundary():
    triangle = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    tolerance = 1e-9
    for gap, collision in (
        (0., True),
        (np.nextafter(tolerance, 0.), True),
        (tolerance, True),
        (np.nextafter(tolerance, np.inf), False),
    ):
        shifted = triangle + [0., 0., gap]
        assert triangles_intersect(triangle, shifted, tolerance) is collision
        assert triangles_intersect(shifted, triangle, tolerance) is collision


def test_triangle_intersection_preserves_fully_degenerate_bounds_fallback():
    point = np.zeros((3, 3))
    tolerance = 1e-9
    assert triangles_intersect(point, point + [tolerance, 0., 0.], tolerance)
    assert not triangles_intersect(
        point, point + [np.nextafter(tolerance, np.inf), 0., 0.], tolerance,
    )


def test_triangle_mesh_intersection_uses_exact_narrow_phase():
    first = np.asarray(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]],
        ]
    )
    overlapping_bounds_but_clear = np.asarray(
        [[[1.5, 1.5, 0.0], [2.5, 1.5, 0.0], [1.5, 2.5, 0.0]]]
    )
    touching = np.asarray(
        [[[1.0, 0.5, -1.0], [1.0, 0.5, 1.0], [1.0, 1.5, 0.0]]]
    )

    assert not triangle_meshes_intersect(first, overlapping_bounds_but_clear)
    assert triangle_meshes_intersect(first, touching)


def test_mesh_components_preserve_exact_vertices_and_component_order():
    from erc_phase1_solution.kinematics import _mesh_components

    mesh = np.asarray([
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
        [[-0., 0., 0.], [-1., 0., 0.], [0., -1., 0.]],
        [[np.nextafter(0., 1.), 0., 0.], [2., 0., 0.], [2., 1., 0.]],
    ])
    components = _mesh_components(mesh)
    assert len(components) == 2
    np.testing.assert_array_equal(components[0], mesh[:2])
    np.testing.assert_array_equal(components[1], mesh[2:])
    assert _mesh_components(np.empty((0, 3, 3))) == ()


def test_triangle_mesh_filtered_broad_phase_matches_brute_force():
    """Whole-mesh facet filtering must not change surface intersections."""

    generator = np.random.default_rng(20260905)
    for tolerance in (0.0, 1e-7, 0.00075):
        for _ in range(30):
            first = generator.uniform(-1.0, 1.0, size=(5, 3, 3))
            second = generator.uniform(-1.0, 1.0, size=(7, 3, 3))
            second += generator.uniform(-1.5, 1.5, size=(1, 1, 3))
            brute_force = any(
                triangles_intersect(left, right, tolerance)
                for left in first
                for right in second
            )

            assert triangle_meshes_intersect(
                first,
                second,
                tolerance,
            ) is brute_force


def test_triangle_mesh_intersection_detects_watertight_containment():
    outer_vertices = np.asarray(
        [
            (x, y, z)
            for x in (-2.0, 2.0)
            for y in (-2.0, 2.0)
            for z in (-2.0, 2.0)
        ]
    )
    outer_faces = np.asarray(
        [
            (0, 1, 3), (0, 3, 2),
            (4, 6, 7), (4, 7, 5),
            (0, 4, 5), (0, 5, 1),
            (2, 3, 7), (2, 7, 6),
            (0, 2, 6), (0, 6, 4),
            (1, 5, 7), (1, 7, 3),
        ]
    )
    outer = outer_vertices[outer_faces]
    inner = np.asarray(
        [[[-0.25, -0.25, 0.0], [0.25, -0.25, 0.0], [0.0, 0.25, 0.0]]]
    )

    # Surface-only testing cannot see a mesh wholly inside another mesh.
    assert not triangle_meshes_intersect(outer, inner)
    assert triangle_meshes_intersect(
        outer,
        inner,
        first_watertight=True,
    )
    assert triangle_meshes_intersect(
        inner,
        outer,
        second_watertight=True,
    )


def test_triangle_mesh_containment_checks_each_disconnected_component():
    outer_vertices = np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    )
    outer_faces = np.asarray(
        [
            (0, 1, 3), (0, 3, 2),
            (4, 6, 7), (4, 7, 5),
            (0, 4, 5), (0, 5, 1),
            (2, 3, 7), (2, 7, 6),
            (0, 2, 6), (0, 6, 4),
            (1, 5, 7), (1, 7, 3),
        ]
    )
    candidate = np.asarray(
        [
            [[3.0, 0.0, 0.0], [3.2, 0.0, 0.0], [3.0, 0.2, 0.0]],
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
        ]
    )

    assert triangle_meshes_intersect(
        outer_vertices[outer_faces],
        candidate,
        first_watertight=True,
    )


def test_triangle_mesh_containment_checks_overlapping_closed_components():
    faces = np.asarray(
        [
            (0, 1, 3), (0, 3, 2),
            (4, 6, 7), (4, 7, 5),
            (0, 4, 5), (0, 5, 1),
            (2, 3, 7), (2, 7, 6),
            (0, 2, 6), (0, 6, 4),
            (1, 5, 7), (1, 7, 3),
        ]
    )

    def box_surface(center_x):
        vertices = np.asarray(
            [
                (center_x + x, y, z)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ]
        )
        return vertices[faces]

    overlapping_shells = np.concatenate(
        (box_surface(-0.25), box_surface(0.25))
    )
    inside_both = np.asarray(
        [[[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.1, 0.0]]]
    )

    assert triangle_meshes_intersect(
        overlapping_shells,
        inside_both,
        first_watertight=True,
    )


def test_triangle_mesh_watertight_flag_does_not_create_aabb_false_positive():
    faces = np.asarray([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)])
    first_vertices = np.asarray(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)]
    )
    clear_vertices = np.asarray(
        [(2.0, 2.0, 2.0), (0.75, 2.0, 2.0), (2.0, 0.75, 2.0), (2.0, 2.0, 0.75)]
    )

    assert not triangle_meshes_intersect(
        first_vertices[faces],
        clear_vertices[faces],
        first_watertight=True,
        second_watertight=True,
    )


def test_joint_interpolation_bounds_component_and_total_arm_motion():
    start = np.zeros(8)
    end = np.asarray([0.2, 1.2, -0.8, 0.4, -0.2, 0.3, -0.1, 0.5])

    waypoints = interpolate_joint_waypoints(
        start,
        end,
        first_arm_index=1,
        maximum_joint_step=0.40,
        orientation_distance=lambda first, last: np.sum(
            np.abs(last[1:] - first[1:])
        ),
        maximum_orientation_step=0.45,
    )

    assert np.allclose(waypoints[-1], end)
    route = (start, *waypoints)
    for first, last in zip(route, route[1:]):
        delta = np.abs(last[1:] - first[1:])
        assert np.max(delta) <= 0.40 + 1e-12
        assert np.sum(delta) <= 0.45 + 1e-12


def test_half_step_joint_interpolation_exposes_the_leg_midpoint():
    start = np.zeros(8)
    end = start.copy()
    end[1] = 0.40

    waypoints = interpolate_joint_waypoints(
        start,
        end,
        first_arm_index=1,
        maximum_joint_step=0.20,
        orientation_distance=lambda first, last: 0.0,
        maximum_orientation_step=0.225,
    )

    assert len(waypoints) == 2
    assert waypoints[0][1] == pytest.approx(0.20)
    assert np.allclose(waypoints[-1], end)


def test_retry_19_home_sweep_hits_robot_but_compact_transport_is_clear(
    official_chain,
    official_right_chain,
):
    collision_links = (
        'base_link',
        'torso_base_link',
        'torso_lift_link',
        *(f'arm_left_{index}_link' for index in range(1, 8)),
        *(f'arm_right_{index}_link' for index in range(1, 8)),
    )
    collision_meshes = _collision_meshes(collision_links)
    assert {mesh.link for mesh in collision_meshes} == set(collision_links)

    front = np.asarray([0.692739, -0.062894, 1.583684])
    grasp = front + np.asarray([0.060, 0.0, -0.015])
    grasp_solution, _ = official_chain.solve(
        pose_matrix(grasp, shelf_pinch_orientations(grasp[2])[0]),
        [np.concatenate(([0.35], OFFER[1:]))],
        position_tolerance=0.012,
        orientation_tolerance=0.10,
        max_iterations=180,
        fixed_positions={'torso_lift_joint': 0.35},
    )
    assert grasp_solution is not None
    half_extents = 0.5 * np.asarray([0.16, 0.03, 0.25])
    centre = front + np.asarray([half_extents[0], 0.0, 0.0])
    inflated = half_extents + 0.015
    world_at_grasp = centre + np.asarray(
        [
            (x * inflated[0], y * inflated[1], z * inflated[2])
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    )
    grasp_transform = official_chain.forward(grasp_solution)
    attached_corners = (
        world_at_grasp - grasp_transform[:3, 3]
    ) @ grasp_transform[:3, :3]

    lowered = np.asarray(
        [
            0.35,
            0.835162,
            0.678417,
            0.054839,
            -2.143497,
            0.440922,
            1.838198,
            -0.546955,
        ]
    )
    legacy_transition = np.asarray(
        [0.35, 0.36, 0.678417, 0.47, -2.35, 0.0, -1.20, 0.0]
    )

    def colliding_links(solution):
        transforms = official_chain.link_transforms(solution)
        right_solution = np.concatenate(
            ([solution[0]], RIGHT_HOME)
        )
        right_transforms = official_right_chain.link_transforms(right_solution)
        transforms.update(
            {
                link: right_transforms[link]
                for link in collision_links
                if link.startswith('arm_right_')
            }
        )
        grasp_pose = official_chain.forward(solution)
        carried_corners = (
            attached_corners @ grasp_pose[:3, :3].T
            + grasp_pose[:3, 3]
        )
        hits = []
        for mesh in collision_meshes:
            transform = transforms[mesh.link]
            world_triangles = (
                mesh.triangles @ transform[:3, :3].T
                + transform[:3, 3]
            )
            if oriented_box_intersects_triangles(
                carried_corners,
                world_triangles,
            ):
                hits.append(mesh.link)
        return hits

    directly_connected = {
        frozenset(('base_link', 'torso_base_link')),
        frozenset(('torso_base_link', 'torso_lift_link')),
        frozenset(('torso_lift_link', 'arm_left_1_link')),
        frozenset(('torso_lift_link', 'arm_right_1_link')),
        *(
            frozenset(
                (f'arm_{side}_{index}_link', f'arm_{side}_{index + 1}_link')
            )
            for side in ('left', 'right')
            for index in range(1, 7)
        ),
    }

    def self_colliding_pair(solution):
        transforms = official_chain.link_transforms(solution)
        right_solution = np.concatenate(([solution[0]], RIGHT_HOME))
        right_transforms = official_right_chain.link_transforms(right_solution)
        transforms.update(
            {
                link: right_transforms[link]
                for link in collision_links
                if link.startswith('arm_right_')
            }
        )
        grouped = {}
        for mesh in collision_meshes:
            transform = transforms[mesh.link]
            grouped.setdefault(mesh.link, []).append(
                mesh.triangles @ transform[:3, :3].T + transform[:3, 3]
            )
        surfaces = {
            link: np.concatenate(link_meshes)
            for link, link_meshes in grouped.items()
        }
        for first_index, first_link in enumerate(collision_links):
            for second_link in collision_links[first_index + 1:]:
                if frozenset((first_link, second_link)) in directly_connected:
                    continue
                if triangle_meshes_intersect(
                    surfaces[first_link],
                    surfaces[second_link],
                ):
                    return first_link, second_link
        return None

    swept_sample = lowered + 0.55 * (legacy_transition - lowered)
    assert colliding_links(swept_sample)

    lowered_transport = lowered.copy()
    lowered_transport[0] = HOME[0]
    assert colliding_links(lowered_transport) == []

    # The top-row compact endpoint keeps gravity on the lower finger while
    # drawing both the book and arm inside the mobile-base envelope.  Its
    # compensated swept route is exercised by the manipulation-planner tests.
    assert self_colliding_pair(SUPPORTED_CARRY) is None
    assert colliding_links(SUPPORTED_CARRY) == []
    compact_pose = official_chain.forward(SUPPORTED_CARRY)
    assert abs(float(compact_pose[2, 1])) >= 0.82
    local_gravity = compact_pose[:3, :3].T @ np.asarray([0.0, 0.0, -1.0])
    support_normal = abs(float(local_gravity[1]))
    support_tangent = float(np.linalg.norm(local_gravity[[0, 2]]))
    assert support_tangent / support_normal < 0.9
    compact_corners = (
        attached_corners @ compact_pose[:3, :3].T
        + compact_pose[:3, 3]
    )
    assert np.max(np.linalg.norm(compact_corners[:, :2], axis=1)) < 0.45
    compact_transforms = official_chain.link_transforms(SUPPORTED_CARRY)
    compact_right_transforms = official_right_chain.link_transforms(
        np.concatenate(([SUPPORTED_CARRY[0]], RIGHT_HOME))
    )
    compact_transforms.update(
        {
            link: compact_right_transforms[link]
            for link in collision_links
            if link.startswith('arm_right_')
        }
    )
    compact_robot_radius = max(
        float(
            np.max(
                np.linalg.norm(
                    (
                        mesh.triangles.reshape((-1, 3))
                        @ compact_transforms[mesh.link][:3, :3].T
                        + compact_transforms[mesh.link][:3, 3]
                    )[:, :2],
                    axis=1,
                )
            )
        )
        for mesh in collision_meshes
    )
    assert compact_robot_radius <= 0.45


def test_solver_recovers_a_reachable_official_pose(official_chain):
    target = official_chain.forward(OFFER)
    seed = np.clip(
        OFFER + np.asarray([0.02, 0.04, -0.03, 0.02, 0.01, 0.0, -0.03, 0.03]),
        official_chain.lower,
        official_chain.upper,
    )
    solution, _ = official_chain.solve(target, [seed], max_iterations=100)
    assert solution is not None
    assert np.all(solution >= official_chain.lower + 0.01 - 1e-12)
    assert np.all(solution <= official_chain.upper - 0.01 + 1e-12)
    error = official_chain.pose_error(official_chain.forward(solution), target)
    assert np.linalg.norm(error[:3]) <= 0.012
    assert np.linalg.norm(error[3:]) <= 0.10


def test_pose_error_detects_exact_half_turn_orientation():
    current = np.eye(4)
    target = np.eye(4)
    target[:3, :3] = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    )

    error = URDFChain.pose_error(current, target)

    assert np.linalg.norm(error[:3]) == pytest.approx(0.0)
    assert np.linalg.norm(error[3:]) == pytest.approx(np.pi)


def test_solver_can_hold_torso_fixed_for_cartesian_arm_paths(official_chain):
    target = official_chain.forward(OFFER)
    seed = HOME.copy()
    solution, _ = official_chain.solve(
        target,
        [seed, OFFER],
        max_iterations=180,
        fixed_positions={'torso_lift_joint': OFFER[0]},
    )

    assert solution is not None
    assert solution[0] == pytest.approx(OFFER[0], abs=1e-12)
    error = official_chain.pose_error(official_chain.forward(solution), target)
    assert np.linalg.norm(error[:3]) <= 0.012
    assert np.linalg.norm(error[3:]) <= 0.10


def test_solver_rejects_invalid_fixed_joint(official_chain):
    with pytest.raises(ValueError, match='inactive joint'):
        official_chain.solve(
            official_chain.forward(HOME),
            [HOME],
            fixed_positions={'missing_joint': 0.0},
        )

    with pytest.raises(ValueError, match='cannot be negative'):
        official_chain.solve(
            official_chain.forward(HOME),
            [HOME],
            joint_limit_margin=-0.01,
        )


@pytest.mark.parametrize('height', [0.605, 0.935, 1.265, 1.595])
def test_nominal_v1_0_3_book_rows_have_an_ik_solution(official_chain, height):
    # Centers of the four active rows in simulation.launch.py at the nominal
    # 0.80 m grasp standoff. Runtime tries the same bounded orientation set.
    solutions = []
    for rotation in shelf_grasp_orientations(height):
        solution, _ = official_chain.solve(
            pose_matrix((0.80, 0.0, height), rotation),
            [PREGRASP, OFFER, HOME],
            max_iterations=180,
        )
        solutions.append(solution)
        if solution is not None:
            break
    assert any(solution is not None for solution in solutions)


def test_top_row_grasp_remains_reachable_with_depth_and_nav_margin(official_chain):
    # Calibrated 0.65 m base standoff plus depth offset and observed margin.
    target = (0.815, -0.016, 1.581)
    solutions = []
    for rotation in shelf_grasp_orientations(target[2]):
        solution, _ = official_chain.solve(
            pose_matrix(target, rotation),
            [PREGRASP, OFFER, HOME],
            max_iterations=180,
        )
        solutions.append(solution)
        if solution is not None:
            break
    assert any(solution is not None for solution in solutions)
