"""Pure collision-policy tests for the rigid-palm preflight helper."""

from __future__ import annotations

import math
from pathlib import Path
import weakref

import numpy as np
import pytest

import erc_phase1_solution.rigid_palm_preflight as preflight_module
from erc_phase1_solution.kinematics import CollisionMesh
from erc_phase1_solution.rigid_palm_preflight import (
    BASE_FINGER_COLLISION_LINKS,
    BookOBB,
    CAGE_FINGERTIP_OVERLAP_CAP_M,
    DEFAULT_COLLISION_PADDING_M,
    FINAL_TANGENT_PALM_OVERLAP_CAP_M,
    FINGER_COLLISION_LINKS,
    FINGERTIP_COLLISION_LINKS,
    GripperCollisionModel,
    GripperSample,
    INNER_OUTER_FINGER_COLLISION_LINKS,
    LEFT_GRIPPER_COLLISION_LINKS,
    PALM_COLLISION_LINK,
    PreflightConfig,
    ProbePhase,
    SUPPORTED_PALM_OVERLAP_CAP_M,
    load_left_gripper_collision_model,
    maximum_tool_vertex_displacement,
    padded_oriented_box_corners,
    preflight_gripper_sweep,
    required_sweep_segments,
    target_obb_axis_projection_overlap,
)


NOW = 20.0
TARGET = 'target_book'
OTHER = 'other_book'


def _box_triangles(half_extent: float = 0.001) -> np.ndarray:
    vertices = np.asarray(
        [
            (x, y, z)
            for x in (-half_extent, half_extent)
            for y in (-half_extent, half_extent)
            for z in (-half_extent, half_extent)
        ],
        dtype=float,
    )
    faces = np.asarray(
        [
            (0, 1, 3),
            (0, 3, 2),
            (4, 6, 7),
            (4, 7, 5),
            (0, 4, 5),
            (0, 5, 1),
            (2, 3, 7),
            (2, 7, 6),
            (0, 2, 6),
            (0, 6, 4),
            (1, 5, 7),
            (1, 7, 3),
        ],
        dtype=int,
    )
    return vertices[faces]


def _model(*, vertex_offset: float = 0.0) -> GripperCollisionModel:
    triangles = _box_triangles() + np.asarray(
        [vertex_offset, 0.0, 0.0],
        dtype=float,
    )
    bounds = np.asarray(
        [
            np.min(triangles, axis=(0, 1)),
            np.max(triangles, axis=(0, 1)),
        ]
    )
    return GripperCollisionModel(
        tuple(
            CollisionMesh(link, triangles.copy(), bounds.copy(), True)
            for link in LEFT_GRIPPER_COLLISION_LINKS
        )
    )


def _transform(position) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, 3] = np.asarray(position, dtype=float)
    return result


def _transforms(
    *,
    contact_link: str | None = None,
    contact_position=(0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    transforms = {
        link: _transform((10.0 + index, 0.0, 0.0))
        for index, link in enumerate(LEFT_GRIPPER_COLLISION_LINKS)
    }
    if contact_link is not None:
        transforms[contact_link] = _transform(contact_position)
    return transforms


def _book(name: str, center, *, observed_at: float = NOW) -> BookOBB:
    pose = _transform(center)
    return BookOBB.from_pose(
        name,
        pose,
        (0.01, 0.01, 0.01),
        observed_at,
    )


def _book_with_x_overlap(name: str, overlap_m: float) -> BookOBB:
    tool_positive_face_x = 0.001
    book_half_extent_x = 0.01
    center_x = tool_positive_face_x + book_half_extent_x - overlap_m
    return _book(name, (center_x, 0.0, 0.0))


def _sample(
    transforms=None,
    *,
    phase: ProbePhase = ProbePhase.APPROACH,
    measured: float = 0.069,
    expected: float = 0.069,
    observed_at: float = NOW,
) -> GripperSample:
    return GripperSample(
        transforms=_transforms() if transforms is None else transforms,
        measured_aperture_m=measured,
        expected_aperture_m=expected,
        observed_at=observed_at,
        phase=phase,
    )


def _run(
    sample: GripperSample,
    *,
    books=None,
    shelf=None,
    expected_books=(TARGET,),
    config=PreflightConfig(),
):
    if books is None:
        books = {TARGET: _book(TARGET, (100.0, 0.0, 0.0))}
    if shelf is None:
        shelf = _box_triangles() + np.asarray([200.0, 0.0, 0.0])
    return preflight_gripper_sweep(
        model=_model(),
        samples=[sample],
        shelf_triangles=shelf,
        books=books,
        target_book=TARGET,
        expected_book_names=expected_books,
        reference_time=NOW,
        config=config,
    )


def test_official_loader_contains_all_nine_collision_bearing_links():
    source_root = Path(__file__).resolve().parents[2]
    urdf = source_root / 'erc_description' / 'urdf' / 'tiago_pro.urdf'
    package_paths = {
        'pal_pro_gripper_description': (
            source_root / 'pal_pro_gripper' / 'pal_pro_gripper_description'
        )
    }

    model = load_left_gripper_collision_model(
        urdf,
        package_paths.__getitem__,
    )

    assert len(LEFT_GRIPPER_COLLISION_LINKS) == 9
    assert {mesh.link for mesh in model.meshes} == set(
        LEFT_GRIPPER_COLLISION_LINKS
    )
    assert all(
        model.meshes_for_link(link)
        for link in LEFT_GRIPPER_COLLISION_LINKS
    )


def test_model_rejects_a_missing_collision_link():
    with pytest.raises(ValueError, match='missing links'):
        GripperCollisionModel(_model().meshes[:-1])


def test_tool_shelf_contact_is_always_rejected():
    result = _run(
        _sample(
            _transforms(contact_link=PALM_COLLISION_LINK),
            phase=ProbePhase.CAGE,
        ),
        shelf=_box_triangles(),
    )

    assert not result.safe
    assert result.code == 'tool_shelf_collision'
    assert result.link == PALM_COLLISION_LINK


def test_tool_non_target_contact_is_always_rejected():
    books = {
        TARGET: _book(TARGET, (100.0, 0.0, 0.0)),
        OTHER: _book(OTHER, (0.0, 0.0, 0.0)),
    }
    result = _run(
        _sample(
            _transforms(contact_link=PALM_COLLISION_LINK),
            phase=ProbePhase.CAGE,
        ),
        books=books,
        expected_books=(TARGET, OTHER),
    )

    assert not result.safe
    assert result.code == 'tool_non_target_collision'
    assert result.obstacle == OTHER


def test_target_palm_contact_is_rejected_during_approach():
    result = _run(
        _sample(
            _transforms(contact_link=PALM_COLLISION_LINK),
            phase=ProbePhase.APPROACH,
        ),
        books={TARGET: _book(TARGET, (0.0, 0.0, 0.0))},
    )

    assert not result.safe
    assert result.code == 'target_palm_contact_outside_support_phase'


@pytest.mark.parametrize(
    ('phase', 'cap_m'),
    (
        (ProbePhase.FINAL_TANGENT, FINAL_TANGENT_PALM_OVERLAP_CAP_M),
        (ProbePhase.LIFT, SUPPORTED_PALM_OVERLAP_CAP_M),
        (ProbePhase.CAGE, SUPPORTED_PALM_OVERLAP_CAP_M),
    ),
)
def test_target_palm_contact_at_phase_cap_is_allowed(phase, cap_m):
    result = _run(
        _sample(
            _transforms(contact_link=PALM_COLLISION_LINK),
            phase=phase,
        ),
        books={TARGET: _book_with_x_overlap(TARGET, cap_m)},
    )

    assert result.safe, result


@pytest.mark.parametrize(
    ('phase', 'cap_m'),
    (
        (ProbePhase.FINAL_TANGENT, FINAL_TANGENT_PALM_OVERLAP_CAP_M),
        (ProbePhase.LIFT, SUPPORTED_PALM_OVERLAP_CAP_M),
        (ProbePhase.CAGE, SUPPORTED_PALM_OVERLAP_CAP_M),
    ),
)
def test_target_palm_contact_above_phase_cap_is_rejected(phase, cap_m):
    result = _run(
        _sample(
            _transforms(contact_link=PALM_COLLISION_LINK),
            phase=phase,
        ),
        books={TARGET: _book_with_x_overlap(TARGET, cap_m + 0.00001)},
    )

    assert not result.safe
    assert result.code == 'target_palm_overlap_exceeds_phase_cap'
    assert f'{cap_m:.9f} m cap' in result.detail


def test_padding_only_palm_proximity_does_not_compute_overlap(monkeypatch):
    def unexpected_overlap(*args, **kwargs):
        pytest.fail('projection overlap requires exact nominal contact')

    monkeypatch.setattr(
        preflight_module,
        'target_obb_axis_projection_overlap',
        unexpected_overlap,
    )
    target_center = 0.001 + 0.01 + 0.00060
    result = _run(
        _sample(
            _transforms(contact_link=PALM_COLLISION_LINK),
            phase=ProbePhase.FINAL_TANGENT,
        ),
        books={TARGET: _book(TARGET, (target_center, 0.0, 0.0))},
    )

    assert result.safe, result


@pytest.mark.parametrize(
    'phase',
    (ProbePhase.DEPARTURE, ProbePhase.TANGENT_APPROACH),
)
def test_proximity_phase_allows_only_padding_near_target_palm(phase):
    # Nominal tool/book surfaces retain a 0.60 mm gap.  The conservative
    # 0.75 mm padded target nevertheless overlaps the palm mesh.
    target_center = 0.001 + 0.01 + 0.00060
    result = _run(
        _sample(
            _transforms(contact_link=PALM_COLLISION_LINK),
            phase=phase,
        ),
        books={TARGET: _book(TARGET, (target_center, 0.0, 0.0))},
    )

    assert result.safe, result


@pytest.mark.parametrize(
    'phase',
    (ProbePhase.DEPARTURE, ProbePhase.TANGENT_APPROACH),
)
def test_proximity_phase_rejects_nominal_target_palm_contact(phase):
    result = _run(
        _sample(
            _transforms(contact_link=PALM_COLLISION_LINK),
            phase=phase,
        ),
        books={TARGET: _book(TARGET, (0.0, 0.0, 0.0))},
    )

    assert not result.safe
    assert result.code == 'target_nominal_palm_contact_during_proximity'


def test_departure_does_not_allow_padded_target_finger_proximity():
    link = FINGER_COLLISION_LINKS[0]
    target_center = 0.001 + 0.01 + 0.00060
    result = _run(
        _sample(
            _transforms(contact_link=link),
            phase=ProbePhase.DEPARTURE,
        ),
        books={TARGET: _book(TARGET, (target_center, 0.0, 0.0))},
    )

    assert not result.safe
    assert result.code == 'target_finger_contact_before_cage'
    assert result.link == link


@pytest.mark.parametrize(
    'phase',
    (ProbePhase.APPROACH, ProbePhase.FINAL_TANGENT, ProbePhase.LIFT),
)
def test_target_finger_contact_is_rejected_before_cage(phase):
    link = FINGER_COLLISION_LINKS[-1]
    result = _run(
        _sample(_transforms(contact_link=link), phase=phase),
        books={TARGET: _book(TARGET, (0.0, 0.0, 0.0))},
    )

    assert not result.safe
    assert result.code == 'target_finger_contact_before_cage'
    assert result.link == link


@pytest.mark.parametrize('link', INNER_OUTER_FINGER_COLLISION_LINKS)
def test_inner_outer_target_contact_is_uncapped_during_cage(link):
    result = _run(
        _sample(
            _transforms(contact_link=link),
            phase=ProbePhase.CAGE,
        ),
        books={TARGET: _book(TARGET, (0.0, 0.0, 0.0))},
    )

    assert result.safe, result


def test_inner_outer_cage_contact_does_not_use_overlap_metric(monkeypatch):
    def unexpected_overlap(*args, **kwargs):
        pytest.fail('articulated finger contact must not use projection cap')

    monkeypatch.setattr(
        preflight_module,
        'target_obb_axis_projection_overlap',
        unexpected_overlap,
    )
    link = INNER_OUTER_FINGER_COLLISION_LINKS[0]
    result = _run(
        _sample(
            _transforms(contact_link=link),
            phase=ProbePhase.CAGE,
        ),
        books={TARGET: _book(TARGET, (0.0, 0.0, 0.0))},
    )

    assert result.safe, result


@pytest.mark.parametrize('link', BASE_FINGER_COLLISION_LINKS)
def test_base_finger_target_contact_is_forbidden_during_cage(link):
    result = _run(
        _sample(
            _transforms(contact_link=link),
            phase=ProbePhase.CAGE,
        ),
        books={TARGET: _book(TARGET, (0.0, 0.0, 0.0))},
    )

    assert not result.safe
    assert result.code == 'target_base_finger_contact_forbidden'
    assert result.link == link


@pytest.mark.parametrize('link', FINGERTIP_COLLISION_LINKS)
def test_fingertip_target_contact_at_cage_cap_is_allowed(link):
    book = _book_with_x_overlap(
        TARGET,
        CAGE_FINGERTIP_OVERLAP_CAP_M,
    )
    result = _run(
        _sample(
            _transforms(contact_link=link),
            phase=ProbePhase.CAGE,
        ),
        books={TARGET: book},
    )

    assert result.safe, result
    overlap = target_obb_axis_projection_overlap(
        book.corners,
        _box_triangles(),
    )
    assert overlap == pytest.approx(CAGE_FINGERTIP_OVERLAP_CAP_M)


@pytest.mark.parametrize('link', FINGERTIP_COLLISION_LINKS)
def test_fingertip_target_contact_above_cage_cap_is_rejected(link):
    result = _run(
        _sample(
            _transforms(contact_link=link),
            phase=ProbePhase.CAGE,
        ),
        books={
            TARGET: _book_with_x_overlap(
                TARGET,
                CAGE_FINGERTIP_OVERLAP_CAP_M + 0.00001,
            )
        },
    )

    assert not result.safe
    assert result.code == 'target_fingertip_overlap_exceeds_cage_cap'
    assert result.link == link


def test_padding_rejects_near_non_target_book():
    # Tool and book surfaces are 0.00060 m apart: geometrically clear without
    # padding, but inside the default 0.00075 m safety envelope.
    separation_center = 0.001 + 0.01 + 0.00060
    books = {
        TARGET: _book(TARGET, (100.0, 0.0, 0.0)),
        OTHER: _book(OTHER, (separation_center, 0.0, 0.0)),
    }
    result = _run(
        _sample(_transforms(contact_link=PALM_COLLISION_LINK)),
        books=books,
        expected_books=(TARGET, OTHER),
    )

    assert not result.safe
    assert result.code == 'tool_non_target_collision'
    assert np.max(
        padded_oriented_box_corners(
            books[OTHER].corners,
            DEFAULT_COLLISION_PADDING_M,
        )[:, 0]
    ) > 0.0


def test_disjoint_book_aabbs_skip_obb_narrow_phase(monkeypatch):
    def unexpected_narrow_phase(*args, **kwargs):
        pytest.fail('disjoint book AABBs must skip OBB SAT')

    monkeypatch.setattr(
        preflight_module,
        'oriented_box_intersects_triangles',
        unexpected_narrow_phase,
    )
    books = {
        TARGET: _book(TARGET, (100.0, 0.0, 0.0)),
        **{
            f'other_{index}': _book(
                f'other_{index}',
                (120.0 + index, 0.0, 0.0),
            )
            for index in range(19)
        },
    }

    result = _run(
        _sample(),
        books=books,
        expected_books=tuple(books),
    )

    assert result.safe, result


def test_book_aabb_candidate_still_runs_obb_collision_check(monkeypatch):
    calls = 0
    actual_narrow_phase = (
        preflight_module.oriented_box_intersects_triangles
    )

    def counted_narrow_phase(*args, **kwargs):
        nonlocal calls
        calls += 1
        return actual_narrow_phase(*args, **kwargs)

    monkeypatch.setattr(
        preflight_module,
        'oriented_box_intersects_triangles',
        counted_narrow_phase,
    )
    result = _run(
        _sample(_transforms(contact_link=PALM_COLLISION_LINK)),
        books={TARGET: _book(TARGET, (0.0, 0.0, 0.0))},
    )

    assert not result.safe
    assert result.code == 'target_palm_contact_outside_support_phase'
    assert calls == 1


def test_streamed_collision_preserves_sample_index():
    first = _transforms(contact_link=PALM_COLLISION_LINK)
    second = {
        link: transform.copy()
        for link, transform in first.items()
    }
    second[PALM_COLLISION_LINK][0, 3] += 0.0003
    # At sample zero the padded target starts 0.20 mm beyond the palm.  The
    # second sample moves only 0.30 mm and enters the padded envelope.
    target_center = 0.001 + 0.01 + 0.00075 + 0.00020

    result = preflight_gripper_sweep(
        model=_model(),
        samples=[
            _sample(first, observed_at=NOW - 0.01),
            _sample(second, observed_at=NOW),
        ],
        shelf_triangles=_box_triangles() + np.asarray([200.0, 0.0, 0.0]),
        books={TARGET: _book(TARGET, (target_center, 0.0, 0.0))},
        target_book=TARGET,
        expected_book_names=(TARGET,),
        reference_time=NOW,
    )

    assert not result.safe
    assert result.code == 'target_palm_contact_outside_support_phase'
    assert result.sample_index == 1


def test_missing_and_nonfinite_link_transforms_fail_closed():
    missing = _transforms()
    missing.pop(FINGER_COLLISION_LINKS[0])
    missing_result = _run(_sample(missing))

    nonfinite = _transforms()
    nonfinite[PALM_COLLISION_LINK][0, 3] = math.nan
    nonfinite_result = _run(_sample(nonfinite))

    assert not missing_result.safe
    assert missing_result.code == 'missing_transform'
    assert not nonfinite_result.safe
    assert nonfinite_result.code == 'invalid_transform'


@pytest.mark.parametrize(
    ('measured', 'expected', 'code'),
    (
        (math.nan, 0.069, 'nonfinite_aperture'),
        (0.069, math.inf, 'nonfinite_aperture'),
        (0.066, 0.069, 'aperture_drift'),
    ),
)
def test_nonfinite_or_drifted_aperture_fails_closed(measured, expected, code):
    result = _run(_sample(measured=measured, expected=expected))

    assert not result.safe
    assert result.code == code


def test_stale_sample_and_stale_book_fail_closed():
    stale_sample = _run(_sample(observed_at=NOW - 0.251))
    stale_book = _run(
        _sample(),
        books={TARGET: _book(TARGET, (100.0, 0.0, 0.0), observed_at=19.0)},
    )

    assert not stale_sample.safe
    assert stale_sample.code == 'stale_input'
    assert not stale_book.safe
    assert stale_book.code == 'stale_input'


def test_missing_book_obb_and_nonfinite_book_pose_fail_closed():
    missing = _run(
        _sample(),
        expected_books=(TARGET, OTHER),
    )
    bad_book = _book(TARGET, (100.0, 0.0, 0.0))
    bad_book.corners[0, 0] = math.nan
    invalid = _run(_sample(), books={TARGET: bad_book})

    assert not missing.safe
    assert missing.code == 'missing_book_obb'
    assert not invalid.safe
    assert invalid.code == 'invalid_book_pose'


def test_sweep_density_uses_rotating_mesh_vertices_not_link_origin():
    model = _model(vertex_offset=0.10)
    first = _transforms()
    second = {link: transform.copy() for link, transform in first.items()}
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    second[PALM_COLLISION_LINK][:3, :3] = rotation

    displacement = maximum_tool_vertex_displacement(model, first, second)
    segments = required_sweep_segments(model, first, second)
    result = preflight_gripper_sweep(
        model=model,
        samples=[
            _sample(first, observed_at=NOW - 0.01),
            _sample(second, observed_at=NOW),
        ],
        shelf_triangles=_box_triangles() + np.asarray([200.0, 0.0, 0.0]),
        books={TARGET: _book(TARGET, (100.0, 0.0, 0.0))},
        target_book=TARGET,
        expected_book_names=(TARGET,),
        reference_time=NOW,
    )

    assert np.allclose(
        first[PALM_COLLISION_LINK][:3, 3],
        second[PALM_COLLISION_LINK][:3, 3],
    )
    assert displacement > 0.10
    assert segments > 1
    assert not result.safe
    assert result.code == 'undersampled_sweep'


def test_dense_clear_sweep_passes():
    model = _model()
    start = _transforms()
    middle = {link: transform.copy() for link, transform in start.items()}
    end = {link: transform.copy() for link, transform in start.items()}
    for transform in middle.values():
        transform[2, 3] += 0.0004
    for transform in end.values():
        transform[2, 3] += 0.0008

    result = preflight_gripper_sweep(
        model=model,
        samples=[
            _sample(start, observed_at=NOW - 0.02),
            _sample(middle, observed_at=NOW - 0.01),
            _sample(end, observed_at=NOW),
        ],
        shelf_triangles=_box_triangles() + np.asarray([200.0, 0.0, 0.0]),
        books={TARGET: _book(TARGET, (100.0, 0.0, 0.0))},
        target_book=TARGET,
        expected_book_names=(TARGET,),
        reference_time=NOW,
    )

    assert result.safe, result


def test_preflight_builds_each_sample_world_surface_once(monkeypatch):
    model = _model()
    start = _transforms()
    middle = {link: transform.copy() for link, transform in start.items()}
    end = {link: transform.copy() for link, transform in start.items()}
    for transform in middle.values():
        transform[2, 3] += 0.0004
    for transform in end.values():
        transform[2, 3] += 0.0008

    calls = 0
    peak_live_surface_maps = 0
    surface_map_references = []
    actual_builder = preflight_module.world_gripper_surfaces

    class TrackedSurfaceMap(dict):
        pass

    def counted_builder(*args, **kwargs):
        nonlocal calls, peak_live_surface_maps
        calls += 1
        result = TrackedSurfaceMap(actual_builder(*args, **kwargs))
        surface_map_references.append(weakref.ref(result))
        live_count = sum(
            reference() is not None
            for reference in surface_map_references
        )
        peak_live_surface_maps = max(
            peak_live_surface_maps,
            live_count,
        )
        return result

    monkeypatch.setattr(
        preflight_module,
        'world_gripper_surfaces',
        counted_builder,
    )
    result = preflight_gripper_sweep(
        model=model,
        samples=[
            _sample(start, observed_at=NOW - 0.02),
            _sample(middle, observed_at=NOW - 0.01),
            _sample(end, observed_at=NOW),
        ],
        shelf_triangles=_box_triangles() + np.asarray([200.0, 0.0, 0.0]),
        books={TARGET: _book(TARGET, (100.0, 0.0, 0.0))},
        target_book=TARGET,
        expected_book_names=(TARGET,),
        reference_time=NOW,
    )

    assert result.safe, result
    assert calls == 3
    assert peak_live_surface_maps <= 2


def test_invalid_sampling_config_fails_closed():
    result = _run(
        _sample(),
        config=PreflightConfig(
            collision_padding_m=0.0005,
            max_vertex_step_m=0.0006,
        ),
    )

    assert not result.safe
    assert result.code == 'invalid_config'
