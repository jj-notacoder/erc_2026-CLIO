"""Current official payload geometry and synthetic adaptive-close regressions.

The synthetic contact windows exercise production evidence gates. They are not
physical retention evidence; nominal URDF mimic FK is not measured finger FK.
"""

from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

pytest.importorskip('rclpy')
from ament_index_python.packages import get_package_share_directory
from erc_phase1_solution.adaptive_grasp import ForceSample, GripperFeedback
from erc_phase1_solution.manipulation_node import ManipulationNode
from erc_phase1_solution.shelf_cradle_geometry import ShelfCradleGeometry


@pytest.fixture(scope='module')
def profile():
    package = Path(__file__).resolve().parents[1]
    return yaml.safe_load((package / 'config/solution.yaml').read_text())[
        'erc_manipulation']['ros__parameters']


@pytest.fixture(scope='module')
def nominal_geometry():
    description = Path(__file__).resolve().parents[2] / 'erc_description'
    model = ShelfCradleGeometry(description / 'urdf/tiago_pro.urdf',
                               get_package_share_directory)
    size = ET.parse(description / 'models/book/sdf/erc_book.sdf').find(
        './/collision/geometry/box/size').text
    dimensions = np.asarray([float(value) for value in size.split()])

    def gap(q):
        surfaces = model.local_surfaces(q)
        left = surfaces['gripper_left_fingertip_left_link']
        right = surfaces['gripper_left_fingertip_right_link']
        return float(right[:, :, 1].min() - left[:, :, 1].max())

    return dimensions, gap


def test_current_default_profile_reaches_nominal_book_contact_and_bounded_preload(
    profile, nominal_geometry,
):
    dimensions, gap = nominal_geometry
    declared = {}
    ManipulationNode._declare_parameters(SimpleNamespace(
        declare_parameter=lambda name, value: declared.update({name: value})))
    names = ('gripper_preclose_position', 'gripper_preload_position',
             'gripper_transport_lock_position', 'grasp_min_position',
             'grasp_min_margin', 'carried_book_dimensions')
    assert {name: profile[name] for name in names} == {
        name: declared[name] for name in names}
    np.testing.assert_allclose(profile['carried_book_dimensions'], dimensions[[2, 1, 0]])
    thickness = float(dimensions[1])
    assert thickness == pytest.approx(.020)
    floor = profile['grasp_min_position'] + profile['grasp_min_margin']
    lock = profile['gripper_transport_lock_position']
    assert 0 <= lock < floor < profile['gripper_preload_position'] < profile[
        'gripper_preclose_position'] < profile['grasp_max_position']
    assert gap(floor) < thickness < gap(profile['gripper_preclose_position'])
    lo, hi = floor, profile['gripper_preclose_position']
    for _ in range(40):
        mid = (lo + hi) / 2
        if gap(mid) < thickness:
            lo = mid
        else:
            hi = mid
    nominal_contact = (lo + hi) / 2
    # Allow the full bounded preload after nominal contact; q is not jaw gap.
    assert lock < nominal_contact - profile['adaptive_gripper_preload_distance']
    assert nominal_contact != pytest.approx(thickness, abs=.001)
    assert gap(.0285 + .001) > thickness  # The former floor cannot acquire it.


def synthetic_close(profile, gap, thickness, mode):
    """Exercise real close/evidence methods with commands replaced by samples."""
    node = object.__new__(ManipulationNode)
    parameter_attributes = {
        'adaptive_close_step': 'adaptive_gripper_close_step',
        'adaptive_preload_distance': 'adaptive_gripper_preload_distance',
        'adaptive_confirmation_seconds': 'adaptive_gripper_confirmation_seconds',
        'adaptive_contact_force_minimum': 'adaptive_gripper_contact_force_minimum',
        'adaptive_contact_force_maximum': 'adaptive_gripper_contact_force_maximum',
        'adaptive_contact_samples': 'adaptive_gripper_contact_samples',
        'adaptive_contact_max_age': 'adaptive_gripper_contact_max_age_seconds',
        'adaptive_contact_max_gap': 'adaptive_gripper_contact_max_gap_seconds',
        'adaptive_contact_min_span': 'adaptive_gripper_contact_min_span_seconds',
        'adaptive_contact_max_skew': 'adaptive_gripper_contact_max_skew_seconds',
        'adaptive_velocity_tolerance': 'adaptive_gripper_velocity_tolerance',
        'adaptive_effort_maximum': 'adaptive_gripper_effort_maximum',
        'adaptive_effort_delta_maximum': 'adaptive_gripper_effort_delta_maximum',
        'adaptive_endpoint_tolerance': 'adaptive_gripper_endpoint_tolerance',
        'adaptive_start_tolerance': 'adaptive_gripper_start_tolerance',
        'adaptive_unilateral_travel_limit': 'adaptive_gripper_unilateral_travel_limit',
        'gripper_open': 'gripper_open_position',
        'gripper_transport_lock': 'gripper_transport_lock_position',
        'grasp_min_position': 'grasp_min_position',
        'grasp_min_margin': 'grasp_min_margin',
        'grasp_max_position': 'grasp_max_position',
        'grasp_contact_max_age': 'grasp_contact_max_age_seconds',
    }
    for attribute, parameter in parameter_attributes.items():
        setattr(node, attribute, profile[parameter])
    now = SimpleNamespace(nanoseconds=10_000_000_000)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    node._target_book_model = None
    node._gripper_open_confirmed = True
    node.joints = {'gripper_left_finger_joint': node.gripper_open}
    node._left_target_contact_ns = node._right_target_contact_ns = 0
    node._book_contact_force_samples = {}
    commands, statuses = [], []
    state = {'first_contact': None}
    node._publish_status = lambda event, **fields: statuses.append((event, fields))

    def update_samples():
        q = node.joints['gripper_left_finger_joint']
        touching = gap(q) <= thickness and mode != 'no_contact'
        if touching and state['first_contact'] is None:
            state['first_contact'] = q
        if mode == 'lost_after_preload' and state['first_contact'] is not None:
            touching = touching and q >= state['first_contact'] - 1e-9
        force = profile['adaptive_gripper_contact_force_maximum'] + 1. if mode == 'overload' else .8
        stamps = [now.nanoseconds - 100_000_000,
                  now.nanoseconds - 50_000_000, now.nanoseconds]
        history = tuple(ForceSample(stamp, force) for stamp in stamps) if touching else ()
        if touching:
            node._target_book_model = 'book_col_3_row_2_red'
        node._book_contact_force_samples = (
            {'book_col_3_row_2_red': (history, history)} if history else {})
        node._left_target_contact_ns = node._right_target_contact_ns = (
            now.nanoseconds if touching else 0)
        node._gripper_feedback_samples = (
            GripperFeedback(now.nanoseconds, q, 0., .2),)

    def command(q):
        commands.append(q)
        node.joints['gripper_left_finger_joint'] = q
        now.nanoseconds += 420_000_000
        update_samples()
        return True

    def wait(duration):
        now.nanoseconds += int(duration * 1e9)
        update_samples()
        return True

    node._command_adaptive_gripper_step = command
    node._wait_sim_duration = wait
    update_samples()
    result = node._adaptive_close_for_grasp()
    return node, result, commands, statuses, state


def test_20mm_close_acquires_below_old_floor_and_reverifies_relative_preload(
    profile, nominal_geometry,
):
    dimensions, gap = nominal_geometry
    node, result, commands, statuses, state = synthetic_close(
        profile, gap, dimensions[1], 'bilateral')
    assert result[0]
    assert node._transport_lock_engaged
    assert state['first_contact'] < .0285 + .001
    assert commands[-2] == pytest.approx(state['first_contact'])
    assert commands[-1] == pytest.approx(
        state['first_contact'] - profile['adaptive_gripper_preload_distance'])
    assert min(commands) >= profile['gripper_transport_lock_position']
    assert statuses[-1][1]['reason'] == 'bilateral_pressure_confirmed'


@pytest.mark.parametrize('mode', ['no_contact', 'overload', 'lost_after_preload'])
def test_20mm_profile_still_rejects_absent_excessive_or_lost_pressure(
    profile, nominal_geometry, mode,
):
    dimensions, gap = nominal_geometry
    node, result, commands, statuses, state = synthetic_close(
        profile, gap, dimensions[1], mode)
    assert not result[0]
    assert not node._transport_lock_engaged
    if mode == 'no_contact':
        assert state['first_contact'] is None
        assert commands[-1] == pytest.approx(
            profile['grasp_min_position'] + profile['grasp_min_margin'])
        assert statuses[-1][1]['reason'] == 'bilateral_contact_not_found'
    elif mode == 'overload':
        assert commands[-1] == pytest.approx(state['first_contact'])
        assert 'overload' in statuses[-1][1]['reason']
    else:
        assert commands[-1] < state['first_contact']
        assert statuses[-1][1]['reason'] != 'bilateral_pressure_confirmed'
