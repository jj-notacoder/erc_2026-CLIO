"""Pure-Python tests for evaluator contracts and runtime safety helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from erc_phase1_solution.runtime_utils import (
    ContactEpisodeTracker,
    book_model_name,
    frames_are_synchronized,
    is_intentional_gripper_book_contact,
    is_scored_obstacle_contact,
    is_target_book_bin_contact,
    is_target_book_non_gripper_robot_contact,
    joint_state_cache_key,
    measured_joint_positions,
    minimum_valid_range,
    normalize_contact_pair,
    polyline_length,
    scored_contact_object,
    stamp_to_nanoseconds,
    validate_task_parameters,
)


@pytest.mark.parametrize(
    ('pair', 'colour', 'expected'),
    [
        (
            (
                'book_col_3_row_2_red::book_base_link::base_link_book_collision',
                'tiago_pro::arm_left_4_link::arm_left_4_link_collision',
            ),
            'red',
            True,
        ),
        (
            (
                'tiago_pro::arm_left_4_link::arm_left_4_link_collision',
                'book_col_3_row_2_red::book_base_link::base_link_book_collision',
            ),
            'red',
            True,
        ),
        (
            (
                'book_col_3_row_2_red::book_base_link::base_link_book_collision',
                'tiago_pro::gripper_left_inner_finger_left_link::collision',
            ),
            'red',
            False,
        ),
        (
            (
                'book_col_3_row_2_red::book_base_link::base_link_book_collision',
                (
                    'tiago_pro::arm_left_7_link::'
                    'arm_left_7_link_fixed_joint_lump__'
                    'gripper_left_base_link_collision_1'
                ),
            ),
            'red',
            False,
        ),
        (
            (
                'book_col_3_row_2_red::book_base_link::base_link_book_collision',
                (
                    'tiago_pro::arm_left_7_link::'
                    'arm_left_7_link_fixed_joint_lump__'
                    'gripper_left_base_link_collision_2'
                ),
            ),
            'red',
            True,
        ),
        (
            (
                'book_col_3_row_2_red::book_base_link::base_link_book_collision',
                (
                    'tiago_pro::arm_left_4_link::'
                    'arm_left_7_link_fixed_joint_lump__'
                    'gripper_left_base_link_collision_1'
                ),
            ),
            'red',
            True,
        ),
        (
            (
                'book_col_3_row_2_red::book_base_link::base_link_book_collision',
                'erc_shelf::shelf_link::collision',
            ),
            'red',
            False,
        ),
        (
            (
                'book_col_3_row_2_blue::book_base_link::base_link_book_collision',
                'tiago_pro::arm_left_4_link::arm_left_4_link_collision',
            ),
            'red',
            False,
        ),
        (
            (
                'book_col_3_row_2_red::book_base_link::base_link_book_collision',
                'collection_bin::link::collision',
            ),
            'red',
            False,
        ),
    ],
)
def test_target_book_non_gripper_robot_contact(pair, colour, expected):
    assert (
        is_target_book_non_gripper_robot_contact(
            normalize_contact_pair(*pair),
            colour,
        )
        is expected
    )


def test_concrete_book_identity_rejects_a_different_same_colour_book():
    selected = 'book_col_2_row_1_red'
    selected_contact = normalize_contact_pair(
        f'{selected}::book_base_link::base_link_book_collision',
        'tiago_pro::arm_left_4_link::arm_left_4_link_collision',
    )
    other_contact = normalize_contact_pair(
        'book_col_3_row_2_red::book_base_link::base_link_book_collision',
        'tiago_pro::arm_left_4_link::arm_left_4_link_collision',
    )

    assert book_model_name(selected_contact[0]) == selected
    assert is_target_book_non_gripper_robot_contact(
        selected_contact,
        'red',
        selected,
    )
    assert not is_target_book_non_gripper_robot_contact(
        other_contact,
        'red',
        selected,
    )


@dataclass
class _Stamp:
    sec: int
    nanosec: int


@pytest.mark.parametrize(
    ('column', 'colour', 'expected'),
    [
        ('1', 'red', (1, 'red')),
        (' 5 ', ' BLUE ', (5, 'blue')),
        (3, 'green', (3, 'green')),
    ],
)
def test_task_parameter_validation(column, colour, expected):
    assert validate_task_parameters(column, colour) == expected


@pytest.mark.parametrize('column', ['0', '6', '2.5', '', True, None])
def test_task_parameter_validation_rejects_invalid_columns(column):
    with pytest.raises(ValueError, match='shelf_column_number'):
        validate_task_parameters(column, 'red')


def test_task_parameter_validation_rejects_invalid_colour():
    with pytest.raises(ValueError, match='book_colour'):
        validate_task_parameters('2', 'purple')


def test_frame_stamp_helpers_enforce_skew():
    first = _Stamp(10, 100_000_000)
    second = _Stamp(10, 175_000_000)
    assert stamp_to_nanoseconds(first) == 10_100_000_000
    assert frames_are_synchronized(first, second, 0.075)
    assert not frames_are_synchronized(first, second, 0.074)
    with pytest.raises(ValueError):
        frames_are_synchronized(first, second, -0.1)


def test_minimum_valid_range_ignores_nan_inf_and_out_of_range_values():
    ranges = [math.nan, math.inf, 0.01, 0.35, 1.2, 30.0]
    assert minimum_valid_range(ranges, 0.05, 25.0) == pytest.approx(0.35)
    assert math.isinf(minimum_valid_range([math.nan, math.inf], 0.05, 25.0))


def test_measured_joint_positions_requires_complete_finite_state():
    state = {'right_1': 0.25, 'right_2': '-0.5'}

    assert measured_joint_positions(
        state,
        ('right_1', 'right_2'),
        group='right arm',
    ) == pytest.approx((0.25, -0.5))
    with pytest.raises(RuntimeError, match='without right arm joints'):
        measured_joint_positions(
            {'right_1': 0.25},
            ('right_1', 'right_2'),
            group='right arm',
        )
    with pytest.raises(RuntimeError, match='non-finite right arm joints'):
        measured_joint_positions(
            {'right_1': 0.25, 'right_2': math.nan},
            ('right_1', 'right_2'),
            group='right arm',
        )


def test_collision_cache_key_changes_with_measured_right_arm_state():
    left = (0.3, 0.1, -0.2)
    parked_right = (1.6, 0.7, -0.1)
    measured_right = (1.6, 0.69, -0.1)
    head = (0.0, -0.28)

    parked_key = joint_state_cache_key(left, parked_right, head)
    measured_key = joint_state_cache_key(left, measured_right, head)

    assert parked_key != measured_key
    assert parked_key == joint_state_cache_key(left, parked_right, head)
    with pytest.raises(ValueError, match='must be finite'):
        joint_state_cache_key(left, (math.nan,), head)


def test_polyline_length_validates_and_measures_planar_samples():
    assert polyline_length([]) == 0.0
    assert polyline_length([(0.0, 0.0), (3.0, 4.0), (3.0, 8.0)]) == 9.0
    with pytest.raises(ValueError, match='x and y'):
        polyline_length([(1.0,)])
    with pytest.raises(ValueError, match='finite'):
        polyline_length([(0.0, 0.0), (math.inf, 1.0)])


def test_official_contact_name_classification():
    palm = (
        'tiago_pro::arm_left_7_link::'
        'arm_left_7_link_fixed_joint_lump__'
        'gripper_left_base_link_collision_1'
    )
    gripper_book = normalize_contact_pair(
        'tiago_pro::gripper_left_fingertip_left_link::collision',
        'book_col_2_row_3_blue::book_base_link::base_link_book_collision',
    )
    bin_book = normalize_contact_pair(
        'erc_collection_bin::collection_bin_base_link::base_link_collection_bin_collision',
        'book_col_2_row_3_blue::book_base_link::base_link_book_collision',
    )
    shelf_arm = normalize_contact_pair(
        'tiago_pro::arm_left_4_link::collision',
        'erc_shelf::shelf_base_link::base_link_shelf_collision',
    )
    palm_book = normalize_contact_pair(
        palm,
        'book_col_2_row_3_blue::book_base_link::base_link_book_collision',
    )
    palm_neighbor = normalize_contact_pair(
        palm,
        'book_col_3_row_2_blue::book_base_link::base_link_book_collision',
    )
    palm_shelf = normalize_contact_pair(
        palm,
        'erc_shelf::shelf_base_link::base_link_shelf_collision',
    )

    assert is_intentional_gripper_book_contact(gripper_book, 'blue')
    assert not is_intentional_gripper_book_contact(gripper_book, 'red')
    assert is_intentional_gripper_book_contact(palm_book, 'blue')
    assert not is_target_book_non_gripper_robot_contact(palm_book, 'blue')
    assert not is_intentional_gripper_book_contact(
        palm_neighbor,
        'blue',
        'book_col_2_row_3_blue',
    )
    assert not is_intentional_gripper_book_contact(palm_shelf, 'blue')
    assert is_scored_obstacle_contact(palm_shelf)
    assert is_target_book_bin_contact(bin_book, 'blue', 'book_col_2_row_3_blue')
    assert not is_target_book_bin_contact(bin_book, 'yellow', 'book_col_2_row_3_blue')
    assert is_scored_obstacle_contact(shelf_arm)
    assert scored_contact_object(shelf_arm) == 'erc_shelf'

    generic_bin_book = normalize_contact_pair(
        'base_link_collection_bin_collision', 'base_link_book_collision'
    )
    assert not is_target_book_bin_contact(generic_bin_book, 'blue')
    assert not is_target_book_bin_contact(
        generic_bin_book, 'blue', 'book_col_2_row_3_blue')


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('book_name,expected', [
    ('book_col_3_row_2_red::book_base_link::base_link_book_collision', True),
    ('erc_world::book_col_3_row_2_red::book_base_link::book_collision', True),
    ('book_col_4_row_2_red::book_base_link::book_collision', False),
    ('book_col_3_row_3_red::book_base_link::book_collision', False),
    ('book_col_3_row_2_blue::book_base_link::book_collision', False),
    ('base_link_book_collision', False),
    ('book_base_link::base_link_book_collision', False),
    ('erc_book_red::book_collision', False),
    ('book_col_3_row_2_red', False),
    ('other_book_col_3_row_2_red::book_collision', False),
    ('book_col_3_row_2_red_copy::book_collision', False),
])
def test_bin_contact_requires_exact_target_scope(book_name, expected, reverse):
    pair = ('erc_collection_bin::collection_bin_base_link::bin_collision', book_name)
    if reverse:
        pair = tuple(reversed(pair))

    assert is_target_book_bin_contact(pair, 'red', 'book_col_3_row_2_red') is expected


@pytest.mark.parametrize('target_model', [None, '', 'base_link_book_collision',
                                         'other_book_col_3_row_2_red',
                                         'book_col_3_row_2_red_copy'])
def test_bin_contact_cannot_infer_missing_or_malformed_latched_identity(target_model):
    pair = ('erc_collection_bin::bin_collision',
            'book_col_3_row_2_red::book_base_link::book_collision')

    assert not is_target_book_bin_contact(pair, 'red', target_model)


def test_contact_episode_tracker_applies_separation_and_cooldown():
    tracker = ContactEpisodeTracker(separation_gap=0.25, cooldown=1.0)
    shelf = ('', 'erc_shelf')

    assert tracker.observe([shelf], 0.0) == {shelf}
    assert tracker.observe([shelf], 0.1) == set()
    assert tracker.active_pairs(0.2) == {shelf}
    assert tracker.active_pairs(0.4) == set()

    # Contact broke, but the one-second same-object cooldown has not elapsed.
    assert tracker.observe([shelf], 0.6) == set()
    assert tracker.observe([shelf], 0.7) == set()
    assert tracker.observe([shelf], 1.2) == {shelf}


def test_multiple_robot_links_coalesce_to_one_scored_object():
    first = normalize_contact_pair(
        'tiago_pro::arm_left_3_link::collision',
        'erc_shelf::shelf_base_link::base_link_shelf_collision',
    )
    second = normalize_contact_pair(
        'tiago_pro::arm_left_4_link::collision',
        'erc_shelf::shelf_base_link::base_link_shelf_collision',
    )
    keys = {
        ('', contacted)
        for contacted in (scored_contact_object(first), scored_contact_object(second))
        if contacted is not None
    }
    assert keys == {('', 'erc_shelf')}
