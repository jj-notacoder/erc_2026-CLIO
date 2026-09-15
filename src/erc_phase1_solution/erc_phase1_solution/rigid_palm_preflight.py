"""Pure collision preflight for the rigid-palm engineering diagnostic.

The live probe deliberately keeps ROS, Gazebo queries, and controller commands
outside this module.  A caller supplies an official gripper collision model,
fresh world-frame link transforms, the static shelf surface, and fresh book
oriented boxes.  The functions here then fail closed before any live motion.

Every sampled state must include all nine collision-bearing left-gripper links.
Adjacent samples are accepted only when the greatest displacement of any mesh
vertex is no larger than the configured sweep step.  Keeping that step below
the collision padding makes the discrete checks a conservative swept-motion
gate and avoids relying on joint-space deltas alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .kinematics import (
    CollisionMesh,
    load_urdf_collision_meshes,
    oriented_box_from_corners,
    oriented_box_intersects_triangles,
    triangle_meshes_intersect,
)


PALM_COLLISION_LINK = 'gripper_left_base_link'
FINGER_COLLISION_LINKS = (
    'gripper_left_base_finger_left_link',
    'gripper_left_inner_finger_left_link',
    'gripper_left_outer_finger_left_link',
    'gripper_left_fingertip_left_link',
    'gripper_left_base_finger_right_link',
    'gripper_left_inner_finger_right_link',
    'gripper_left_outer_finger_right_link',
    'gripper_left_fingertip_right_link',
)
BASE_FINGER_COLLISION_LINKS = (
    'gripper_left_base_finger_left_link',
    'gripper_left_base_finger_right_link',
)
INNER_OUTER_FINGER_COLLISION_LINKS = (
    'gripper_left_inner_finger_left_link',
    'gripper_left_outer_finger_left_link',
    'gripper_left_inner_finger_right_link',
    'gripper_left_outer_finger_right_link',
)
FINGERTIP_COLLISION_LINKS = (
    'gripper_left_fingertip_left_link',
    'gripper_left_fingertip_right_link',
)
LEFT_GRIPPER_COLLISION_LINKS = (
    PALM_COLLISION_LINK,
    *FINGER_COLLISION_LINKS,
)

DEFAULT_COLLISION_PADDING_M = 0.00075
DEFAULT_MAX_VERTEX_STEP_M = 0.00050
FINAL_TANGENT_PALM_OVERLAP_CAP_M = 0.00010
SUPPORTED_PALM_OVERLAP_CAP_M = 0.00050
CAGE_FINGERTIP_OVERLAP_CAP_M = 0.0010


class ProbePhase(str, Enum):
    """Contact policy phase for a rigid-palm sweep sample."""

    APPROACH = 'approach'
    DEPARTURE = 'departure'
    TANGENT_APPROACH = 'tangent_approach'
    FINAL_TANGENT = 'final_tangent'
    LIFT = 'lift'
    CAGE = 'cage'


@dataclass(frozen=True)
class GripperCollisionModel:
    """All official collision meshes for the nine left-gripper links."""

    meshes: tuple[CollisionMesh, ...]

    def __post_init__(self) -> None:
        meshes = tuple(self.meshes)
        object.__setattr__(self, 'meshes', meshes)
        if not meshes:
            raise ValueError('left-gripper collision model cannot be empty')

        expected = set(LEFT_GRIPPER_COLLISION_LINKS)
        actual = {mesh.link for mesh in meshes}
        missing = expected - actual
        unexpected = actual - expected
        if missing or unexpected:
            details = []
            if missing:
                details.append(f'missing links: {sorted(missing)}')
            if unexpected:
                details.append(f'unexpected links: {sorted(unexpected)}')
            raise ValueError('; '.join(details))

        for mesh in meshes:
            triangles = _validated_surface(
                mesh.triangles,
                label=f'collision mesh for {mesh.link}',
            )
            bounds = np.asarray(mesh.bounds, dtype=float)
            if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
                raise ValueError(
                    f'collision bounds for {mesh.link} must be finite (2, 3)'
                )
            calculated = np.asarray(
                [
                    np.min(triangles, axis=(0, 1)),
                    np.max(triangles, axis=(0, 1)),
                ]
            )
            if not np.allclose(bounds, calculated, atol=1e-9, rtol=0.0):
                raise ValueError(
                    f'collision bounds for {mesh.link} do not match its mesh'
                )

    def meshes_for_link(self, link: str) -> tuple[CollisionMesh, ...]:
        """Return every collision geometry carried by one required link."""
        return tuple(mesh for mesh in self.meshes if mesh.link == link)


@dataclass(frozen=True)
class BookOBB:
    """Fresh world-frame book oriented box in kinematics corner order."""

    name: str
    corners: np.ndarray
    observed_at: float

    @classmethod
    def from_pose(
        cls,
        name: str,
        pose: Sequence[Sequence[float]],
        half_extents: Sequence[float],
        observed_at: float,
    ) -> 'BookOBB':
        """Construct a world OBB from a homogeneous pose and half extents."""
        transform = _validated_transform(pose, label=f'book pose for {name}')
        half = np.asarray(half_extents, dtype=float)
        if (
            half.shape != (3,)
            or not np.all(np.isfinite(half))
            or np.any(half <= 0.0)
        ):
            raise ValueError(
                'book half extents must be three finite positives'
            )
        signs = _ordered_box_signs()
        local = signs * half
        corners = (
            local @ transform[:3, :3].T
            + transform[:3, 3]
        )
        return cls(name=name, corners=corners, observed_at=observed_at)


@dataclass(frozen=True)
class GripperSample:
    """One fresh, sampled gripper state to be checked in the world frame."""

    transforms: Mapping[str, np.ndarray]
    measured_aperture_m: float
    expected_aperture_m: float
    observed_at: float
    phase: ProbePhase


@dataclass(frozen=True)
class PreflightConfig:
    """Conservative geometric and freshness limits for the preflight."""

    collision_padding_m: float = DEFAULT_COLLISION_PADDING_M
    max_vertex_step_m: float = DEFAULT_MAX_VERTEX_STEP_M
    aperture_tolerance_m: float = 0.001
    max_state_age_seconds: float = 0.25
    future_tolerance_seconds: float = 0.02
    rigid_transform_tolerance: float = 1e-5


@dataclass(frozen=True)
class PreflightResult:
    """Fail-closed result suitable for logging immediately before motion."""

    safe: bool
    code: str
    detail: str
    sample_index: int | None = None
    link: str | None = None
    obstacle: str | None = None


class _InvalidInput(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def load_left_gripper_collision_model(
    urdf_path: str | Path,
    package_resolver: Callable[[str], str | Path],
) -> GripperCollisionModel:
    """Load and validate all nine official left-gripper collision links."""
    meshes = load_urdf_collision_meshes(
        urdf_path,
        LEFT_GRIPPER_COLLISION_LINKS,
        package_resolver,
    )
    return GripperCollisionModel(meshes=meshes)


def padded_oriented_box_corners(
    corners: Sequence[Sequence[float]],
    padding_m: float = DEFAULT_COLLISION_PADDING_M,
) -> np.ndarray:
    """Expand an OBB by ``padding_m`` along each of its local axes."""
    padding = float(padding_m)
    if not math.isfinite(padding) or padding < 0.0:
        raise ValueError('OBB padding must be finite and nonnegative')
    points = np.asarray(corners, dtype=float)
    if not np.all(np.isfinite(points)):
        raise ValueError('oriented-box corners must be finite')
    center, axes, half_extents = oriented_box_from_corners(points)
    local = _ordered_box_signs() * (half_extents + padding)
    return local @ axes.T + center


def target_obb_axis_projection_overlap(
    target_corners: Sequence[Sequence[float]],
    tool_triangles: Sequence[Sequence[Sequence[float]]],
) -> float:
    """Conservatively bound contact overlap along the target OBB axes.

    Call this only after an exact nominal target/mesh intersection test.  The
    minimum interval overlap across the target's three axes is an upper bound
    on a full separating-axis penetration depth because mesh-face and
    edge-cross axes are deliberately omitted.  That conservative bias is
    appropriate for the shallow palm cap and the simple fingertip geometry,
    but not for the articulated inner/outer mechanisms whose long projected
    spans make this estimate unrelated to their local surface contact.
    """
    corners = np.asarray(target_corners, dtype=float)
    if not np.all(np.isfinite(corners)):
        raise ValueError('target OBB corners must be finite')
    center, axes, half_extents = oriented_box_from_corners(corners)
    surface = _validated_surface(
        tool_triangles,
        label='tool contact surface',
    )
    target_center = center @ axes
    target_lower = target_center - half_extents
    target_upper = target_center + half_extents
    projections = surface.reshape((-1, 3)) @ axes
    tool_lower = np.min(projections, axis=0)
    tool_upper = np.max(projections, axis=0)
    overlap = np.minimum(target_upper, tool_upper) - np.maximum(
        target_lower,
        tool_lower,
    )
    # The exact contact predicate includes a 1 nm numerical tolerance.  A
    # tangent result can therefore produce a tiny negative projection here.
    return max(0.0, float(np.min(overlap)))


def world_gripper_surfaces(
    model: GripperCollisionModel,
    transforms: Mapping[str, np.ndarray],
    *,
    rigid_transform_tolerance: float = 1e-5,
) -> dict[str, np.ndarray]:
    """Transform and group every left-gripper collision surface into world."""
    validated = _validated_link_transforms(
        transforms,
        rigid_transform_tolerance=rigid_transform_tolerance,
    )
    grouped: dict[str, list[np.ndarray]] = {
        link: [] for link in LEFT_GRIPPER_COLLISION_LINKS
    }
    for mesh in model.meshes:
        transform = validated[mesh.link]
        grouped[mesh.link].append(
            mesh.triangles @ transform[:3, :3].T + transform[:3, 3]
        )
    return {
        link: parts[0] if len(parts) == 1 else np.concatenate(parts)
        for link, parts in grouped.items()
    }


def maximum_tool_vertex_displacement(
    model: GripperCollisionModel,
    first_transforms: Mapping[str, np.ndarray],
    second_transforms: Mapping[str, np.ndarray],
    *,
    rigid_transform_tolerance: float = 1e-5,
) -> float:
    """Return the greatest world displacement of any gripper mesh vertex."""
    first = world_gripper_surfaces(
        model,
        first_transforms,
        rigid_transform_tolerance=rigid_transform_tolerance,
    )
    second = world_gripper_surfaces(
        model,
        second_transforms,
        rigid_transform_tolerance=rigid_transform_tolerance,
    )
    return _maximum_world_surface_vertex_displacement(first, second)


def _maximum_world_surface_vertex_displacement(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
) -> float:
    """Compare corresponding vertices in two validated world surfaces."""
    maximum = 0.0
    for link in LEFT_GRIPPER_COLLISION_LINKS:
        if link not in first or link not in second:
            raise ValueError(f'world surfaces missing {link!r}')
        if first[link].shape != second[link].shape:
            raise ValueError(
                f'world surface shape changed for {link!r}'
            )
        displacement = np.linalg.norm(
            second[link] - first[link],
            axis=2,
        )
        maximum = max(maximum, float(np.max(displacement)))
    return maximum


def required_sweep_segments(
    model: GripperCollisionModel,
    first_transforms: Mapping[str, np.ndarray],
    second_transforms: Mapping[str, np.ndarray],
    *,
    max_vertex_step_m: float = DEFAULT_MAX_VERTEX_STEP_M,
    rigid_transform_tolerance: float = 1e-5,
) -> int:
    """Return a lower bound on segments from endpoint mesh displacement.

    The caller must still generate FK transforms at each resulting interior
    state and pass all of them to :func:`preflight_gripper_sweep`; nonlinear
    link motion can require further subdivision.
    """
    step = float(max_vertex_step_m)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError('maximum vertex step must be finite and positive')
    displacement = maximum_tool_vertex_displacement(
        model,
        first_transforms,
        second_transforms,
        rigid_transform_tolerance=rigid_transform_tolerance,
    )
    return max(1, int(math.ceil(displacement / step)))


def preflight_gripper_sweep(
    *,
    model: GripperCollisionModel,
    samples: Sequence[GripperSample],
    shelf_triangles: Sequence[Sequence[Sequence[float]]],
    books: Mapping[str, BookOBB],
    target_book: str,
    expected_book_names: Iterable[str],
    reference_time: float,
    config: PreflightConfig = PreflightConfig(),
) -> PreflightResult:
    """Check a sampled gripper sweep against shelf and every known book.

    Contact policy is intentionally narrow:

    * shelf and non-target-book contact is always forbidden;
    * target contact is forbidden throughout ``APPROACH``;
    * ``DEPARTURE`` and ``TANGENT_APPROACH`` permit only palm proximity
      created by padding while the nominal target OBB remains geometrically
      disjoint;
    * palm contact is capped at 0.10 mm during ``FINAL_TANGENT`` and at
      0.50 mm during ``LIFT`` and ``CAGE``;
    * ``CAGE`` permits uncapped target contact on the articulated inner/outer
      mechanisms and up to 1.0 mm on the simpler fingertips, while the two
      base-finger links remain forbidden.

    Any missing, stale, nonfinite, non-rigid, aperture-drifted, or undersampled
    input returns an unsafe result instead of attempting a partial check.
    """
    try:
        validated_config = _validated_config(config)
        now = float(reference_time)
        if not math.isfinite(now):
            raise _InvalidInput(
                'invalid_reference_time',
                'reference time must be finite',
            )
        if not samples:
            raise _InvalidInput('missing_samples', 'sweep has no samples')
        if not target_book:
            raise _InvalidInput(
                'missing_target_book',
                'target book name is empty',
            )

        shelf = _validated_surface(shelf_triangles, label='shelf surface')
        checked_books = _validated_books(
            books,
            target_book=target_book,
            expected_book_names=expected_book_names,
            reference_time=now,
            config=validated_config,
        )

        padded_books = tuple(
            (
                name,
                padded_oriented_box_corners(
                    book.corners,
                    validated_config.collision_padding_m,
                ),
                np.asarray(book.corners, dtype=float),
            )
            for name, book in checked_books.items()
        )
        padded_book_bounds = np.asarray(
            [
                _surface_bounds(padded_corners)
                for _, padded_corners, _ in padded_books
            ],
            dtype=float,
        )
        shelf_bounds = _surface_bounds(shelf)
        watertight = {
            link: all(
                mesh.watertight
                for mesh in model.meshes_for_link(link)
            )
            for link in LEFT_GRIPPER_COLLISION_LINKS
        }

        previous_time: float | None = None
        previous_surfaces: dict[str, np.ndarray] | None = None
        first_collision: PreflightResult | None = None
        sample_count = 0
        for index, sample in enumerate(samples):
            phase = _validated_sample(
                sample,
                index=index,
                previous_time=previous_time,
                reference_time=now,
                config=validated_config,
            )
            surfaces = world_gripper_surfaces(
                model,
                sample.transforms,
                rigid_transform_tolerance=(
                    validated_config.rigid_transform_tolerance
                ),
            )
            if previous_surfaces is not None:
                displacement = _maximum_world_surface_vertex_displacement(
                    previous_surfaces,
                    surfaces,
                )
                if (
                    displacement
                    > validated_config.max_vertex_step_m + 1e-12
                ):
                    raise _InvalidInput(
                        'undersampled_sweep',
                        'sample '
                        f'{index - 1}->{index} moves a tool vertex '
                        f'{displacement:.9f} m, above the '
                        f'{validated_config.max_vertex_step_m:.9f} m limit',
                    )
            if first_collision is None:
                first_collision = _sample_collision_result(
                    sample_index=index,
                    phase=phase,
                    surfaces=surfaces,
                    shelf=shelf,
                    shelf_bounds=shelf_bounds,
                    padded_books=padded_books,
                    padded_book_bounds=padded_book_bounds,
                    target_book=target_book,
                    watertight=watertight,
                    padding_m=validated_config.collision_padding_m,
                )
            previous_surfaces = surfaces
            previous_time = float(sample.observed_at)
            sample_count += 1
    except (_InvalidInput, TypeError, ValueError) as error:
        code = (
            error.code
            if isinstance(error, _InvalidInput)
            else 'invalid_input'
        )
        return PreflightResult(False, code, str(error))

    if first_collision is not None:
        return first_collision
    return PreflightResult(
        True,
        'clear',
        f'{sample_count} sampled states passed geometric preflight',
    )


def _sample_collision_result(
    *,
    sample_index: int,
    phase: ProbePhase,
    surfaces: Mapping[str, np.ndarray],
    shelf: np.ndarray,
    shelf_bounds: np.ndarray,
    padded_books: Sequence[tuple[str, np.ndarray, np.ndarray]],
    padded_book_bounds: np.ndarray,
    target_book: str,
    watertight: Mapping[str, bool],
    padding_m: float,
) -> PreflightResult | None:
    """Return the first phase-policy collision for one world sample."""
    for link in LEFT_GRIPPER_COLLISION_LINKS:
        surface = surfaces[link]
        surface_bounds = _surface_bounds(surface)
        shelf_candidate = _aabbs_overlap(
            surface_bounds,
            shelf_bounds,
            tolerance=padding_m,
        )
        if shelf_candidate and triangle_meshes_intersect(
            surface,
            shelf,
            tolerance=padding_m,
            first_watertight=watertight[link],
        ):
            return PreflightResult(
                False,
                'tool_shelf_collision',
                f'{link} intersects the padded shelf envelope',
                sample_index,
                link,
                'shelf',
            )

        book_candidates = _overlapping_aabb_indices(
            surface_bounds,
            padded_book_bounds,
            # Preserve the narrow phase's inclusive 1 nm tolerance at an
            # otherwise exactly separated floating-point boundary.
            tolerance=1e-9,
        )
        for book_index in book_candidates:
            name, padded_corners, nominal_corners = padded_books[
                int(book_index)
            ]
            if not oriented_box_intersects_triangles(
                padded_corners,
                surface,
                closed_surface=watertight[link],
            ):
                continue
            if name != target_book:
                return PreflightResult(
                    False,
                    'tool_non_target_collision',
                    f'{link} intersects non-target book {name}',
                    sample_index,
                    link,
                    name,
                )
            violation = _target_contact_violation(
                sample_index=sample_index,
                phase=phase,
                link=link,
                target_name=name,
                nominal_corners=nominal_corners,
                surface=surface,
                surface_watertight=watertight[link],
            )
            if violation is not None:
                return violation
    return None


def _target_contact_violation(
    *,
    sample_index: int,
    phase: ProbePhase,
    link: str,
    target_name: str,
    nominal_corners: np.ndarray,
    surface: np.ndarray,
    surface_watertight: bool,
) -> PreflightResult | None:
    """Apply the phase policy after padded target contact is confirmed."""

    def nominal_contact() -> bool:
        return oriented_box_intersects_triangles(
            nominal_corners,
            surface,
            closed_surface=surface_watertight,
        )

    def excessive_overlap(
        cap_m: float,
        *,
        code: str,
        contact_kind: str,
    ) -> PreflightResult | None:
        # Padding-only proximity is permitted in phases that permit this link;
        # the penetration metric is meaningful only after nominal contact.
        if not nominal_contact():
            return None
        overlap = target_obb_axis_projection_overlap(
            nominal_corners,
            surface,
        )
        if overlap <= cap_m + 1e-12:
            return None
        return PreflightResult(
            False,
            code,
            f'{contact_kind} target overlap {overlap:.9f} m exceeds '
            f'the {cap_m:.9f} m cap during {phase.value}',
            sample_index,
            link,
            target_name,
        )

    if phase in {
        ProbePhase.DEPARTURE,
        ProbePhase.TANGENT_APPROACH,
    } and link == PALM_COLLISION_LINK:
        if not nominal_contact():
            return None
        code = 'target_nominal_palm_contact_during_proximity'
    elif link == PALM_COLLISION_LINK and phase is ProbePhase.FINAL_TANGENT:
        return excessive_overlap(
            FINAL_TANGENT_PALM_OVERLAP_CAP_M,
            code='target_palm_overlap_exceeds_phase_cap',
            contact_kind='palm',
        )
    elif link == PALM_COLLISION_LINK and phase in {
        ProbePhase.LIFT,
        ProbePhase.CAGE,
    }:
        return excessive_overlap(
            SUPPORTED_PALM_OVERLAP_CAP_M,
            code='target_palm_overlap_exceeds_phase_cap',
            contact_kind='palm',
        )
    elif phase is ProbePhase.CAGE and link in BASE_FINGER_COLLISION_LINKS:
        return PreflightResult(
            False,
            'target_base_finger_contact_forbidden',
            f'{link} has forbidden padded target contact during cage',
            sample_index,
            link,
            target_name,
        )
    elif (
        phase is ProbePhase.CAGE
        and link in INNER_OUTER_FINGER_COLLISION_LINKS
    ):
        # Their articulated, elongated surfaces project across much of a book;
        # the OBB-axis metric substantially overstates local penetration.
        return None
    elif phase is ProbePhase.CAGE and link in FINGERTIP_COLLISION_LINKS:
        return excessive_overlap(
            CAGE_FINGERTIP_OVERLAP_CAP_M,
            code='target_fingertip_overlap_exceeds_cage_cap',
            contact_kind='fingertip',
        )
    else:
        code = (
            'target_finger_contact_before_cage'
            if link in FINGER_COLLISION_LINKS
            else 'target_palm_contact_outside_support_phase'
        )
    return PreflightResult(
        False,
        code,
        f'{link} has forbidden target contact during {phase.value}',
        sample_index,
        link,
        target_name,
    )


def _ordered_box_signs() -> np.ndarray:
    return np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=float,
    )


def _validated_surface(
    triangles: Sequence[Sequence[Sequence[float]]],
    *,
    label: str,
) -> np.ndarray:
    surface = np.asarray(triangles, dtype=float)
    if surface.ndim != 3 or surface.shape[1:] != (3, 3) or not len(surface):
        raise ValueError(f'{label} must be a nonempty (N, 3, 3) array')
    if not np.all(np.isfinite(surface)):
        raise ValueError(f'{label} coordinates must be finite')
    return surface


def _surface_bounds(vertices: np.ndarray) -> np.ndarray:
    """Return world AABB bounds for triangles or ordered OBB corners."""
    points = np.asarray(vertices, dtype=float)
    axes = tuple(range(points.ndim - 1))
    return np.asarray(
        [np.min(points, axis=axes), np.max(points, axis=axes)],
        dtype=float,
    )


def _aabbs_overlap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    tolerance: float = 0.0,
) -> bool:
    """Return whether two validated ``(lower, upper)`` AABBs overlap."""
    return bool(
        np.all(first[1] + tolerance >= second[0])
        and np.all(second[1] + tolerance >= first[0])
    )


def _overlapping_aabb_indices(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Vector-filter obstacle AABBs before the relatively costly OBB SAT."""
    if not len(candidates):
        return np.empty(0, dtype=int)
    overlaps = (
        np.all(query[1] + tolerance >= candidates[:, 0], axis=1)
        & np.all(candidates[:, 1] + tolerance >= query[0], axis=1)
    )
    return np.flatnonzero(overlaps)


def _validated_transform(
    transform: Sequence[Sequence[float]],
    *,
    label: str,
    tolerance: float = 1e-5,
) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f'{label} must be a 4x4 transform')
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f'{label} must be finite')
    if not np.allclose(
        matrix[3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=tolerance,
        rtol=0.0,
    ):
        raise ValueError(f'{label} must have a homogeneous final row')
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=tolerance,
        rtol=0.0,
    ) or not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        abs_tol=tolerance,
        rel_tol=0.0,
    ):
        raise ValueError(f'{label} rotation must be proper and orthonormal')
    return matrix


def _validated_link_transforms(
    transforms: Mapping[str, np.ndarray],
    *,
    rigid_transform_tolerance: float,
) -> dict[str, np.ndarray]:
    if not isinstance(transforms, Mapping):
        raise ValueError('link transforms must be a mapping')
    missing = set(LEFT_GRIPPER_COLLISION_LINKS) - set(transforms)
    if missing:
        raise ValueError(f'link transforms missing: {sorted(missing)}')
    return {
        link: _validated_transform(
            transforms[link],
            label=f'world transform for {link}',
            tolerance=rigid_transform_tolerance,
        )
        for link in LEFT_GRIPPER_COLLISION_LINKS
    }


def _validated_config(config: PreflightConfig) -> PreflightConfig:
    finite_positive = {
        'collision padding': config.collision_padding_m,
        'maximum vertex step': config.max_vertex_step_m,
        'aperture tolerance': config.aperture_tolerance_m,
        'maximum state age': config.max_state_age_seconds,
        'rigid transform tolerance': config.rigid_transform_tolerance,
    }
    for name, raw_value in finite_positive.items():
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            raise _InvalidInput(
                'invalid_config',
                f'{name} must be finite and positive',
            )
    future = float(config.future_tolerance_seconds)
    if not math.isfinite(future) or future < 0.0:
        raise _InvalidInput(
            'invalid_config',
            'future tolerance must be finite and nonnegative',
        )
    if config.max_vertex_step_m > config.collision_padding_m:
        raise _InvalidInput(
            'invalid_config',
            'maximum tool-vertex step must not exceed collision padding',
        )
    return config


def _validate_freshness(
    observed_at: float,
    *,
    label: str,
    reference_time: float,
    config: PreflightConfig,
) -> float:
    stamp = float(observed_at)
    if not math.isfinite(stamp):
        raise _InvalidInput(
            'nonfinite_timestamp',
            f'{label} time is nonfinite',
        )
    age = reference_time - stamp
    if age > config.max_state_age_seconds:
        raise _InvalidInput(
            'stale_input',
            f'{label} is stale by {age:.6f} seconds',
        )
    if age < -config.future_tolerance_seconds:
        raise _InvalidInput(
            'future_input',
            f'{label} is {-age:.6f} seconds in the future',
        )
    return stamp


def _validated_sample(
    sample: GripperSample,
    *,
    index: int,
    previous_time: float | None,
    reference_time: float,
    config: PreflightConfig,
) -> ProbePhase:
    try:
        phase = ProbePhase(sample.phase)
    except (TypeError, ValueError) as error:
        raise _InvalidInput(
            'invalid_phase',
            f'sample {index} has invalid phase {sample.phase!r}',
        ) from error
    measured = float(sample.measured_aperture_m)
    expected = float(sample.expected_aperture_m)
    if not math.isfinite(measured) or not math.isfinite(expected):
        raise _InvalidInput(
            'nonfinite_aperture',
            f'sample {index} aperture must be finite',
        )
    if abs(measured - expected) > config.aperture_tolerance_m:
        raise _InvalidInput(
            'aperture_drift',
            f'sample {index} aperture drift is '
            f'{abs(measured - expected):.9f} m',
        )
    stamp = _validate_freshness(
        sample.observed_at,
        label=f'sample {index}',
        reference_time=reference_time,
        config=config,
    )
    if previous_time is not None and stamp + 1e-12 < previous_time:
        raise _InvalidInput(
            'nonmonotonic_samples',
            f'sample {index} is older than sample {index - 1}',
        )
    try:
        _validated_link_transforms(
            sample.transforms,
            rigid_transform_tolerance=config.rigid_transform_tolerance,
        )
    except (TypeError, ValueError) as error:
        message = str(error)
        code = (
            'missing_transform'
            if 'missing' in message
            else 'invalid_transform'
        )
        raise _InvalidInput(code, f'sample {index}: {message}') from error
    return phase


def _validated_books(
    books: Mapping[str, BookOBB],
    *,
    target_book: str,
    expected_book_names: Iterable[str],
    reference_time: float,
    config: PreflightConfig,
) -> dict[str, BookOBB]:
    if not isinstance(books, Mapping):
        raise _InvalidInput('missing_books', 'book OBBs must be a mapping')
    expected = set(expected_book_names)
    if not expected or target_book not in expected:
        raise _InvalidInput(
            'missing_expected_books',
            'expected books must be nonempty and include the target',
        )
    missing = expected - set(books)
    if missing:
        raise _InvalidInput(
            'missing_book_obb',
            f'book OBBs missing: {sorted(missing)}',
        )
    if target_book not in books:
        raise _InvalidInput(
            'missing_target_book',
            f'target book {target_book!r} has no OBB',
        )

    validated: dict[str, BookOBB] = {}
    for key, book in books.items():
        if not isinstance(book, BookOBB):
            raise _InvalidInput(
                'invalid_book_obb',
                f'book entry {key!r} is not a BookOBB',
            )
        if not key or book.name != key:
            raise _InvalidInput(
                'invalid_book_name',
                f'book mapping key {key!r} does not match {book.name!r}',
            )
        points = np.asarray(book.corners, dtype=float)
        if not np.all(np.isfinite(points)):
            raise _InvalidInput(
                'invalid_book_pose',
                f'book {key!r} OBB has nonfinite coordinates',
            )
        try:
            oriented_box_from_corners(points)
        except ValueError as error:
            raise _InvalidInput(
                'invalid_book_pose',
                f'book {key!r} OBB is invalid: {error}',
            ) from error
        _validate_freshness(
            book.observed_at,
            label=f'book {key!r}',
            reference_time=reference_time,
            config=config,
        )
        validated[key] = book
    return validated
