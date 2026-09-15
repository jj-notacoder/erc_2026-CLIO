"""Pure-Python validation, timing, and contact helpers for runtime nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import struct
from typing import Dict, Iterable, Mapping, Sequence, Set, Tuple


VALID_BOOK_COLOURS = frozenset({'red', 'blue', 'green', 'yellow'})
ContactPair = Tuple[str, str]


def measured_joint_positions(
    joints: Mapping[str, float],
    names: Sequence[str],
    *,
    group: str,
) -> Tuple[float, ...]:
    """Return finite measured joints, failing closed when state is incomplete."""
    missing = [name for name in names if name not in joints]
    if missing:
        raise RuntimeError(
            f'Cannot collision-check without {group} joints: {missing}'
        )
    try:
        positions = tuple(float(joints[name]) for name in names)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f'Cannot collision-check with invalid {group} joints'
        ) from exc
    if not all(math.isfinite(position) for position in positions):
        raise RuntimeError(
            f'Cannot collision-check with non-finite {group} joints'
        )
    return positions


def joint_state_cache_key(
    *groups: Sequence[float],
    decimals: int = 10,
) -> bytes:
    """Return a stable finite-state key for collision-result caches."""
    try:
        positions = tuple(
            round(float(position), decimals)
            for group in groups
            for position in group
        )
    except (TypeError, ValueError) as exc:
        raise ValueError('collision cache state must be numeric') from exc
    if not all(math.isfinite(position) for position in positions):
        raise ValueError('collision cache state must be finite')
    return struct.pack(f'={len(positions)}d', *positions)


def validate_task_parameters(column_value, colour_value) -> tuple[int, str]:
    """Normalize and validate the two evaluator-facing launch parameters."""
    if isinstance(column_value, bool):
        raise ValueError('shelf_column_number must be an integer from 1 to 5')
    try:
        column = int(str(column_value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            'shelf_column_number must be an integer from 1 to 5'
        ) from exc
    if column not in range(1, 6):
        raise ValueError('shelf_column_number must be in the range 1..5')

    colour = str(colour_value).strip().lower()
    if colour not in VALID_BOOK_COLOURS:
        choices = ', '.join(sorted(VALID_BOOK_COLOURS))
        raise ValueError(f'book_colour must be one of: {choices}')
    return column, colour


def stamp_to_nanoseconds(stamp) -> int:
    """Return nanoseconds for a ROS Time-like object with sec/nanosec fields."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def frames_are_synchronized(first_stamp, second_stamp, maximum_skew: float) -> bool:
    """Report whether two ROS Time-like stamps are within the allowed skew."""
    if not math.isfinite(maximum_skew) or maximum_skew < 0.0:
        raise ValueError('maximum_skew must be finite and non-negative')
    difference = abs(
        stamp_to_nanoseconds(first_stamp) - stamp_to_nanoseconds(second_stamp)
    )
    return difference <= int(maximum_skew * 1_000_000_000)


def minimum_valid_range(
    ranges: Iterable[float], range_min: float, range_max: float
) -> float:
    """Return the nearest finite in-range laser sample, or infinity."""
    if not math.isfinite(range_min) or not math.isfinite(range_max):
        return float('inf')
    valid = (
        float(value)
        for value in ranges
        if math.isfinite(value) and range_min <= value <= range_max
    )
    return min(valid, default=float('inf'))


def polyline_length(points: Sequence[Sequence[float]]) -> float:
    """Return the planar length of an ordered x/y point sequence."""
    total = 0.0
    previous = None
    for point in points:
        if len(point) < 2:
            raise ValueError('path points must contain x and y')
        current = float(point[0]), float(point[1])
        if not all(math.isfinite(value) for value in current):
            raise ValueError('path points must be finite')
        if previous is not None:
            total += math.hypot(current[0] - previous[0], current[1] - previous[1])
        previous = current
    return total


def normalize_contact_pair(first: str, second: str) -> ContactPair:
    """Create a deterministic contact-pair key."""
    first_name, second_name = str(first), str(second)
    if first_name <= second_name:
        return first_name, second_name
    return second_name, first_name


def _contains_token(name: str, token: str) -> bool:
    return token in name.lower()


_FIXED_JOINT_LEFT_PALM_COLLISION = (
    'tiago_pro::arm_left_7_link::arm_left_7_link_fixed_joint_lump__'
    'gripper_left_base_link_collision_1'
)


def _is_gripper_collision(name: str) -> bool:
    """Recognize normal scopes and Gazebo's fixed-joint-lumped left palm."""

    lowered = str(name).lower()
    return bool(
        '::gripper_' in lowered
        or lowered.startswith('gripper_')
        or lowered == _FIXED_JOINT_LEFT_PALM_COLLISION
    )


def _is_book(name: str) -> bool:
    lowered = name.lower()
    return any(
        token in lowered
        for token in ('book_col_', 'erc_book', 'book_base_link', 'book_collision')
    )


def _book_matches_colour(name: str, colour: str) -> bool:
    if not _is_book(name):
        return False
    components = re.split(r'[^a-z0-9]+|_', name.lower())
    observed_colours = VALID_BOOK_COLOURS.intersection(components)
    # Some ros_gz versions expose only `base_link_book_collision`. Retain this
    # broad classification for conservative collision/grasp diagnostics; it is
    # not proof of target identity for bin delivery.
    return not observed_colours or colour.lower() in observed_colours


def book_model_name(name: str) -> str | None:
    """Return the concrete spawned book model encoded in a scoped name."""
    match = re.search(
        r'book_col_\d+_row_\d+_(?:red|green|yellow|blue)',
        str(name).lower(),
    )
    return match.group(0) if match is not None else None


def _book_matches_target(
    name: str,
    colour: str,
    target_model: str | None,
) -> bool:
    if not _book_matches_colour(name, colour):
        return False
    observed_model = book_model_name(name)
    if target_model and observed_model:
        return observed_model == str(target_model).lower()
    # Some ros_gz contact paths contain only ``base_link_book_collision``.
    # Preserve the colour fallback for those genuinely anonymous samples.
    return True


def is_scored_obstacle_contact(pair: ContactPair) -> bool:
    """Return true for contacts with objects penalized by the Phase-1 rubric."""
    obstacle_tokens = (
        'erc_shelf',
        'book_col_',
        'erc_book',
        'book_collision',
        'erc_table',
        'collection_bin',
    )
    return any(
        _contains_token(name, token)
        for name in pair
        for token in obstacle_tokens
    )


def scored_contact_object(pair: ContactPair) -> str | None:
    """Return the contacted rubric object, coalescing multiple robot links."""
    for name in pair:
        lowered = name.lower()
        if not any(
            token in lowered
            for token in (
                'erc_shelf',
                'book_col_',
                'erc_book',
                'book_collision',
                'erc_table',
                'collection_bin',
            )
        ):
            continue
        # Gazebo scoped names begin with the spawned model. Counting that scope
        # prevents simultaneous contacts from two robot links being treated as
        # two separate collision episodes with the same object.
        return lowered.split('::', 1)[0]
    return None


def is_intentional_gripper_book_contact(
    pair: ContactPair,
    target_colour: str,
    target_model: str | None = None,
) -> bool:
    """Identify the allowed contact between a gripper and the target book."""
    first, second = pair
    return (
        (
            _is_gripper_collision(first)
            and _book_matches_target(second, target_colour, target_model)
        )
        or (
            _is_gripper_collision(second)
            and _book_matches_target(first, target_colour, target_model)
        )
    )


def is_target_book_non_gripper_robot_contact(
    pair: ContactPair,
    target_colour: str,
    target_model: str | None = None,
) -> bool:
    """Identify an unsafe target-book contact with the TIAGo robot body.

    Finger and gripper contacts are deliberately excluded because they form the
    grasp. The official Gazebo contact bridge preserves the ``tiago_pro``
    model scope for robot collisions, even when it shortens the book name.
    """
    first, second = pair
    for book, other in ((first, second), (second, first)):
        lowered = other.lower()
        if (
            _book_matches_target(book, target_colour, target_model)
            and lowered.startswith('tiago_pro::')
            and not _is_gripper_collision(other)
        ):
            return True
    return False


def is_target_book_bin_contact(
    pair: ContactPair,
    target_colour: str,
    target_model: str | None = None,
) -> bool:
    """Require the exact latched model in a scoped book/bin collision pair.

    Colour-only and anonymous book names cannot establish delivery identity.
    Other contact classifiers retain their conservative compatibility fallback.
    """
    model = str(target_model or '').strip().lower()
    if not model or book_model_name(model) != model:
        return False
    if not _book_matches_colour(model, target_colour):
        return False

    def matches_book_scope(name: str) -> bool:
        components = str(name).lower().split('::')
        # A model name embedded in another identifier, or a bare model without
        # a scoped collision name, is not evidence from that exact book.
        return bool(components[-1] and model in components[:-1])

    first, second = pair
    return (
        'collection_bin' in first.lower()
        and matches_book_scope(second)
    ) or (
        'collection_bin' in second.lower()
        and matches_book_scope(first)
    )


@dataclass
class ContactEpisodeTracker:
    """Deduplicate contact samples into rubric-style collision episodes.

    Contact bridges publish repeated samples while contact remains unbroken. A
    pair is considered separated after ``separation_gap`` without a sample and
    cannot start another counted episode until ``cooldown`` has elapsed.
    """

    separation_gap: float = 0.25
    cooldown: float = 1.0
    _last_seen: Dict[ContactPair, float] = field(default_factory=dict)
    _last_counted: Dict[ContactPair, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.separation_gap <= 0.0 or self.cooldown < 0.0:
            raise ValueError('contact timing values must be positive')

    def observe(
        self, pairs: Iterable[ContactPair], timestamp: float
    ) -> Set[ContactPair]:
        """Record samples and return pairs that begin countable new episodes."""
        if not math.isfinite(timestamp):
            raise ValueError('contact timestamp must be finite')
        started: Set[ContactPair] = set()
        for pair in set(pairs):
            last_seen = self._last_seen.get(pair, float('-inf'))
            last_counted = self._last_counted.get(pair, float('-inf'))
            was_active = timestamp - last_seen <= self.separation_gap
            cooldown_elapsed = timestamp - last_counted >= self.cooldown
            if not was_active and cooldown_elapsed:
                started.add(pair)
                self._last_counted[pair] = timestamp
            self._last_seen[pair] = timestamp
        self._prune(timestamp)
        return started

    def active_pairs(self, timestamp: float) -> Set[ContactPair]:
        """Return pairs sampled recently enough to still be continuous."""
        return {
            pair
            for pair, last_seen in self._last_seen.items()
            if timestamp - last_seen <= self.separation_gap
        }

    def _prune(self, timestamp: float) -> None:
        horizon = max(10.0, 4.0 * self.cooldown)
        stale = [
            pair
            for pair, last_seen in self._last_seen.items()
            if timestamp - last_seen > horizon
        ]
        for pair in stale:
            self._last_seen.pop(pair, None)
            self._last_counted.pop(pair, None)


__all__ = [
    'ContactEpisodeTracker',
    'ContactPair',
    'VALID_BOOK_COLOURS',
    'frames_are_synchronized',
    'is_intentional_gripper_book_contact',
    'is_scored_obstacle_contact',
    'is_target_book_bin_contact',
    'is_target_book_non_gripper_robot_contact',
    'joint_state_cache_key',
    'measured_joint_positions',
    'minimum_valid_range',
    'normalize_contact_pair',
    'polyline_length',
    'scored_contact_object',
    'stamp_to_nanoseconds',
    'validate_task_parameters',
]
