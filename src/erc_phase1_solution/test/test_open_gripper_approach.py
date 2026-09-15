"""Resting-book approach checks, including the observed official20mm geometry."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from erc_phase1_solution import open_gripper_approach as guard
from erc_phase1_solution.kinematics import URDFChain


class TranslationChain:
    active_names = ('arm_left_1_joint',)
    lower = np.asarray([-3.])
    upper = np.asarray([3.])

    def __init__(self):
        self.seen = []

    def forward(self, joints):
        self.seen.append(np.asarray(joints).copy())
        pose = np.eye(4)
        pose[0, 3] = joints[0]
        pose[2, 3] = 1.
        return pose


def fake_node(collision_link=None):
    surfaces = {}
    for link in guard.LEFT_GRIPPER_COLLISION_LINKS:
        # The normal tool is behind the frame origin. For a selected test link
        # place a small triangle in the path of the static book instead.
        x = 0. if link == collision_link else -5.
        surfaces[link] = np.asarray([[[x, -.1, -.1], [x, .1, -.1], [x, 0., .1]]])
    apertures = []

    def local_surfaces(aperture):
        apertures.append(aperture)
        return surfaces

    node = SimpleNamespace(
        chain=TranslationChain(), gripper_open=.069,
        carried_book_dimensions=np.asarray([.16, .02, .25]),
        # An approach guard must not reuse the held-book padding.
        carried_collision_padding=.015,
        _shelf_cradle_geometry=SimpleNamespace(
            local_surfaces=local_surfaces,
            watertight={link: False for link in surfaces},
        ),
    )
    return node, surfaces, apertures


@pytest.mark.parametrize('link', guard.LEFT_GRIPPER_COLLISION_LINKS)
def test_each_of_nine_tool_links_blocks_an_intermediate_collision(link):
    node, _, _ = fake_node(link)
    result = guard.check_open_gripper_approach(node, [1., 0., 1.], [[0.], [2.]])
    assert not result.ok
    assert result.reason == 'open_approach_book_intersection'
    assert result.collision_link == link
    assert result.leg_index == 0 and result.sample_index not in (0, 100)


def test_minimum_sampling_and_maximum_joint_step_are_enforced_for_every_leg():
    node, _, apertures = fake_node()
    result = guard.check_open_gripper_approach(node, [1., 0., 1.], [[0.], [.1], [2.1]])
    assert result.ok
    assert result.samples_checked == 61 + 101
    samples = np.asarray(node.chain.seen)
    assert np.max(np.abs(np.diff(samples, axis=0))) <= .02 + 1e-12
    assert apertures == [.069]  # One cached local geometry for the entire route.


@pytest.mark.parametrize('route,reason', [
    ([], 'joint_shape_invalid'),
    ([[0.]], 'joint_shape_invalid'),
    ([[0., 0.], [0., 0.]], 'joint_shape_invalid'),
    ([[0.], [float('nan')]], 'joint_nonfinite'),
    ([[0.], [float('inf')]], 'joint_nonfinite'),
    ([[0.], [3.01]], 'joint_limit'),
])
def test_bad_route_fails_before_any_fk(route, reason):
    node, _, _ = fake_node()
    result = guard.check_open_gripper_approach(node, [1., 0., 1.], route)
    assert not result.ok and result.reason == 'open_approach_' + reason
    assert not node.chain.seen


@pytest.mark.parametrize('front,dimensions', [
    ([1., 0.], [.16, .02, .25]),
    ([1., 0., float('nan')], [.16, .02, .25]),
    ([1., 0., 1.], [.16, 0., .25]),
    ([1., 0., 1.], [.16, -.02, .25]),
    ([1., 0., 1.], [.16, float('inf'), .25]),
])
def test_bad_book_geometry_fails_closed(front, dimensions):
    node, _, _ = fake_node()
    node.carried_book_dimensions = dimensions
    result = guard.check_open_gripper_approach(node, front, [[0.], [.1]])
    assert not result.ok and result.reason == 'open_approach_book_geometry_invalid'


@pytest.mark.parametrize('aperture', [-.001, .070, float('nan')])
def test_invalid_open_position_is_rejected(aperture):
    node, _, _ = fake_node()
    node.gripper_open = aperture
    result = guard.check_open_gripper_approach(node, [1., 0., 1.], [[0.], [.1]])
    assert not result.ok and result.reason == 'open_approach_aperture_invalid'


@pytest.mark.parametrize('fault', ['missing_mesh', 'nan_mesh', 'empty_mesh', 'bad_topology', 'bad_fk'])
def test_missing_or_malformed_collision_inputs_fail_closed(fault):
    node, surfaces, _ = fake_node()
    if fault == 'missing_mesh':
        surfaces.pop(guard.PALM_COLLISION_LINK)
    elif fault == 'nan_mesh':
        surfaces[guard.PALM_COLLISION_LINK][0, 0, 0] = np.nan
    elif fault == 'empty_mesh':
        surfaces[guard.PALM_COLLISION_LINK] = np.empty((0, 3, 3))
    elif fault == 'bad_topology':
        node._shelf_cradle_geometry.watertight[guard.PALM_COLLISION_LINK] = None
    else:
        node.chain.forward = lambda _: np.zeros((4, 4))
    result = guard.check_open_gripper_approach(node, [1., 0., 1.], [[0.], [.1]])
    assert not result.ok
    assert result.reason == ('open_approach_fk_invalid' if fault == 'bad_fk'
                             else 'open_approach_tool_geometry_invalid')


def test_torso_binding_and_joint_limits_are_validated():
    node, _, _ = fake_node()
    node.chain.active_names = ('torso_lift_joint',)
    result = guard.check_open_gripper_approach(
        node, [1., 0., 1.], [[.35], [.34]], torso_height=.35)
    assert result.reason == 'open_approach_torso_mismatch'
    node.chain.lower = np.asarray([np.nan])
    assert guard.check_open_gripper_approach(
        node, [1., 0., 1.], [[.35], [.35]]).reason == 'open_approach_joint_limits_invalid'


@pytest.fixture(scope='module')
def official_observed_geometry():
    from ament_index_python.packages import get_package_share_directory
    package = Path(__file__).resolve().parents[1]
    fixture = json.loads((package / 'test/data/open_gripper_approach_observed.json').read_text())
    urdf = package.parent / 'erc_description/urdf/tiago_pro.urdf'
    assert hashlib.sha256(urdf.read_bytes()).hexdigest() == fixture['urdf_sha256']
    sdf = package.parent / 'erc_description/models/book/sdf/erc_book.sdf'
    size = ET.parse(sdf).find('.//collision/geometry/box/size').text
    nominal_size = np.asarray([float(value) for value in size.split()])
    np.testing.assert_allclose(fixture['dimensions'], nominal_size[[2, 1, 0]])
    node = SimpleNamespace(
        chain=URDFChain.from_urdf(urdf, 'base_footprint', 'gripper_left_grasping_link'),
        carried_book_dimensions=np.asarray(fixture['dimensions']),
        gripper_open=fixture['gripper_open'], carried_collision_padding=.015,
        _shelf_cradle_geometry=guard.ShelfCradleGeometry(urdf, get_package_share_directory),
    )
    return node, fixture


def test_official_20mm_observed_60mm_grasp_depth_hits_palm(official_observed_geometry):
    node, fixture = official_observed_geometry
    endpoint = fixture['depth_060_endpoint']
    result = guard.check_open_gripper_approach(
        node, fixture['front'], [endpoint, endpoint], torso_height=fixture['torso_height'])
    assert not result.ok
    assert result.reason == 'open_approach_book_intersection'
    assert result.collision_link == guard.PALM_COLLISION_LINK
    assert result.min_palm_front_gap_m < -.02


def test_official_20mm_observed_25mm_depth_entire_sampled_path_is_clear(official_observed_geometry):
    node, fixture = official_observed_geometry
    result = guard.check_open_gripper_approach(
        node, fixture['front'], fixture['depth_025_route'], torso_height=fixture['torso_height'])
    assert result.ok, result
    assert result.samples_checked == 305
    # This real 10.19mm gap is valid in the nominal geometry. Applying the
    # unrelated15mm carried-box inflation here would wrongly reject it.
    assert .010 < result.min_palm_front_gap_m < .011


def test_deeper_extension_is_rejected_at_first_sampled_palm_contact(official_observed_geometry):
    node, fixture = official_observed_geometry
    route = [*fixture['depth_025_route'], fixture['depth_060_endpoint']]
    result = guard.check_open_gripper_approach(
        node, fixture['front'], route, torso_height=fixture['torso_height'])
    assert not result.ok
    assert result.reason == 'open_approach_book_intersection'
    assert result.collision_link == guard.PALM_COLLISION_LINK
    assert result.leg_index == 5 and 0 < result.sample_index < 60
    assert -.001 < result.min_palm_front_gap_m < 0.
