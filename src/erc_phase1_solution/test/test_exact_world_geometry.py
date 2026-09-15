"""Exact-geometry reuse and real scene/body method differential fixtures.

Synthetic small meshes exercise actual geometry expressions and predicates.
The scene differential records the same spy arguments/order as the preexisting
snapshot fixture; it does not claim physical-model or live-context validation.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
from erc_phase1_solution.exact_world_geometry import (
    ExactWorldGeometry, _WorldSurface, ordinary_model_producer,
)
from erc_phase1_solution.kinematics import (
    CollisionMesh, PreparedTriangleMesh, triangle_meshes_intersect,
)
from erc_phase1_solution.sample_collision_snapshot import _capture_scene_robot_snapshot
from test_sample_collision_snapshot import Fixture, Bin, Tool, actual_class, box, signature
from world_geometry_test_support import FIXTURE, FIXTURE_SHA256, restore_world_geometry_source


ROOT = Path(__file__).resolve().parents[1]


def owned(shift=0.):
    mesh = ModelLocalMesh(box(shift))
    return mesh, mesh.snapshot()[0]


@pytest.mark.parametrize('angle', [0., .1, -1.2, np.pi / 2])
@pytest.mark.parametrize('translation', [0., -0., 1e-10, -2.])
def test_dense_expression_bounds_and_repeated_outputs_are_bitwise_identical(angle, translation):
    mesh, surface = owned()
    transform = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    transform[:2, :2] = [[c, -s], [s, c]]
    transform[:3, 3] = [translation, -.23, .42]
    expected = surface @ transform[:3, :3].T + transform[:3, 3]
    bounds = np.asarray([expected.min(axis=(0, 1)), expected.max(axis=(0, 1))])
    cache = ExactWorldGeometry()
    first = cache.capture(mesh, surface, transform)
    assert first is cache.capture(mesh, surface, transform.copy())
    actual, actual_bounds = first.views()
    assert signature(actual) == signature(expected)
    assert signature(actual_bounds) == signature(bounds)
    assert (cache.hits, cache.misses, cache.fallbacks) == (1, 1, 0)
    assert actual is not first.views()[0]
    assert not actual.flags.writeable and not actual_bounds.flags.writeable


def test_exact_transform_bytes_not_joint_rounding_and_mutation_never_reuses_old_geometry():
    mesh, surface = owned()
    transform = np.eye(4)
    cache = ExactWorldGeometry()
    first = cache.capture(mesh, surface, transform)
    before = signature(first.views()[0])
    transform[0, 3] = np.nextafter(0., 1.)
    second = cache.capture(mesh, surface, transform)
    assert second is not first and cache.misses == 2
    transform[0, 3] = .125
    third = cache.capture(mesh, surface, transform)
    assert signature(third.views()[0]) != before
    assert signature(first.views()[0]) == before
    transform[0, 3] = 0.
    assert cache.capture(mesh, surface, transform) is first


def test_equal_geometry_new_owner_is_a_miss_and_old_raw_array_cannot_mutate_owner():
    raw = box()
    first = ModelLocalMesh(raw)
    cache = ExactWorldGeometry()
    a = cache.capture(first, first.snapshot()[0], np.eye(4))
    raw += 10.
    second = ModelLocalMesh(first.snapshot()[0].copy())
    b = cache.capture(second, second.snapshot()[0], np.eye(4))
    assert a is not b and signature(a.views()[0]) == signature(b.views()[0])
    assert cache.misses == 2


def test_fresh_metadata_cannot_corrupt_cache_or_snapshot_bounds():
    mesh, surface = owned()
    value = ExactWorldGeometry().capture(mesh, surface, np.eye(4))
    world, bounds = value.views()
    world.shape = (-1,)
    bounds.shape = (6,)
    assert not value.matches(world)
    assert value.views()[0].shape == surface.shape
    assert value.views()[1].shape == (2, 3)
    with pytest.raises(ValueError):
        value.views()[0].flags.writeable = True
    with pytest.raises(AttributeError):
        value._content = b'changed'


@pytest.mark.parametrize('case', ['raw', 'readonly_raw', 'surface_view', 'partial_surface',
                                  'float32', 'transform_view', 'transform_fortran',
                                  'transform_shape', 'transform_nonfinite', 'subclass'])
def test_unsupported_inputs_fall_back_without_entries(case):
    class Custom(np.ndarray):
        pass
    mesh, surface = owned()
    transform = np.eye(4)
    if case == 'raw': surface = box()
    elif case == 'readonly_raw':
        surface = box(); surface.flags.writeable = False
    elif case == 'surface_view': surface = surface.view(Custom)
    elif case == 'partial_surface': surface = surface[:-1]
    elif case == 'float32': transform = transform.astype(np.float32)
    elif case == 'transform_view': transform = transform.view()
    elif case == 'transform_fortran': transform = np.asfortranarray(transform)
    elif case == 'transform_shape': transform = np.eye(3)
    elif case == 'transform_nonfinite': transform[1, 1] = np.nan
    elif case == 'subclass': transform = transform.view(Custom)
    cache = ExactWorldGeometry()
    assert cache.capture(mesh, surface, transform) is None
    assert not cache._entries and cache._bytes == 0 and cache.fallbacks == 1


@pytest.mark.parametrize('maximum_entries,maximum_bytes', [(True, 1), (0, 1), (1, 0), (1, 1.)])
def test_invalid_capacity_rejected(maximum_entries, maximum_bytes):
    with pytest.raises(ValueError):
        ExactWorldGeometry(maximum_entries=maximum_entries, maximum_bytes=maximum_bytes)


def test_entry_and_byte_caps_count_retained_models_and_evict_lru():
    mesh, surface = owned()
    weight = mesh.nbytes + surface.nbytes + 48 + np.eye(4).nbytes
    cache = ExactWorldGeometry(maximum_entries=2, maximum_bytes=2 * weight)
    transforms = [np.eye(4) for _ in range(3)]
    for i, t in enumerate(transforms): t[0, 3] = i
    a = cache.capture(mesh, surface, transforms[0])
    cache.capture(mesh, surface, transforms[1])
    assert cache.capture(mesh, surface, transforms[0]) is a
    cache.capture(mesh, surface, transforms[2])
    assert len(cache._entries) == 2 and cache._bytes == 2 * weight
    assert cache.capture(mesh, surface, transforms[0]) is a
    cache.capture(mesh, surface, transforms[1])
    assert cache.misses == 4 and cache._bytes <= cache.maximum_bytes
    too_small = ExactWorldGeometry(maximum_bytes=weight - 1)
    assert too_small.capture(mesh, surface, np.eye(4)) is None
    assert too_small._bytes == 0


@pytest.mark.parametrize('shift', [.2, np.nextafter(.2, np.inf), .199, 0.])
def test_touch_near_boundary_intersection_and_containment_match_actual_predicate(shift):
    mesh, surface = owned()
    transform = np.eye(4); transform[0, 3] = shift
    cached = ExactWorldGeometry().capture(mesh, surface, transform).views()[0]
    dense = surface @ transform[:3, :3].T + transform[:3, 3]
    other = box()
    def verdict(value):
        return triangle_meshes_intersect(PreparedTriangleMesh(value), PreparedTriangleMesh(other),
                                        first_watertight=True, second_watertight=True)
    assert verdict(cached) == verdict(dense)


def test_snapshot_reuses_private_bytes_and_original_bounds_without_copying():
    f = Fixture(); cache = ExactWorldGeometry(); robot = {}; geometry = {}
    transforms = f.transforms(f.q, right_positions=f.right, head_positions=f.head)
    for mesh in f.node.carried_collision_meshes:
        value = cache.capture(mesh.local_mesh, mesh.triangles, transforms[mesh.link])
        robot[mesh.link] = value.views()[0]; geometry[mesh.link] = value
    snapshot = _capture_scene_robot_snapshot(f.node, f.q, f.right, f.head, robot,
                                              _world_geometry=geometry)
    assert snapshot is not None
    assert f.call(snapshot) is None
    resolved = snapshot.resolved(f.node, f.q, f.right, f.head)
    for link, shape, content, bounds in snapshot._frozen:
        assert (shape, content, bounds) == geometry[link].frozen()
        assert content is geometry[link].frozen()[1]
        assert bounds is geometry[link].frozen()[2]
        assert signature(resolved[0][link]) == signature(robot[link])
    f.head[0] = 1e-11
    assert snapshot.resolved(f.node, f.q, f.right, f.head) is None


def test_snapshot_rejects_foreign_immutable_bytes_and_metadata_but_retains_owned_fallback():
    f = Fixture(); robot = f.robot(); link = next(iter(robot))
    value = _WorldSurface(robot[link])
    # Identical values without that immutable byte owner are not provenance.
    robot[link] = np.frombuffer(robot[link].tobytes(), dtype=float).reshape(robot[link].shape)
    assert _capture_scene_robot_snapshot(f.node, f.q, f.right, f.head, robot,
                                         _world_geometry={link: value}) is None
    assert f.snapshot() is not None
    robot[link] = value.views()[0]
    robot[link].shape = (-1, 3)
    assert _capture_scene_robot_snapshot(f.node, f.q, f.right, f.head, robot,
                                         _world_geometry={link: value}) is None


@pytest.mark.parametrize('changed', ['world', 'transforms', 'body', 'raw_model'])
def test_custom_producer_admission_falls_back(changed):
    f = Fixture()
    assert ordinary_model_producer(f.node)
    if changed == 'world': f.node._world_collision_surfaces = lambda *a, **k: {}
    elif changed == 'transforms': f.node._collision_link_transforms = lambda *a, **k: {}
    elif changed == 'body': f.node._robot_self_collision = lambda *a, **k: None
    else:
        mesh = f.node.carried_collision_meshes[0]
        f.node.carried_collision_meshes = (CollisionMesh(mesh.link, mesh.triangles.copy(), mesh.bounds, True),)
    assert not ordinary_model_producer(f.node)


def source_for_parent(tmp_path):
    path = tmp_path / 'r57_scene.py'
    current = (ROOT / 'erc_phase1_solution/scene_checked_place.py').read_text()
    path.write_text(restore_world_geometry_source(current, 'scene_checked_place.py'))
    return path


@pytest.mark.parametrize('rejection', [False, True])
@pytest.mark.parametrize('custom', [False, True])
def test_actual_scene_and_body_preserve_full_order_values_context_and_result(tmp_path, rejection, custom):
    parent = source_for_parent(tmp_path)
    rows = []
    for changed in (False, True):
        f = Fixture(rejection=rejection)
        scene_class = f.scene if changed else actual_class(parent, f, scene=True)
        obstacle = type('CustomBin', (Bin,), {})() if custom else Bin()
        checker = scene_class(f.node, obstacle, Tool(), np.ones((8, 3)))
        outcomes = []
        for i in range(4):
            q = f.q.copy(); q[1] = .01 * i
            # Same transforms, different full body-state key, then exact new context.
            if i == 3: f.right[0] = .012; f.head[0] = .013
            result = checker.sample(q, .017, False)
            outcomes.append((result, checker.last_rejection, checker.minimum_moving_left_z,
                             checker.samples, checker.cache_hits))
        rows.append((outcomes, f.events, f.node._self_collision_cache,
                     f.node._static_self_collision_cache, f.world_calls))
        if changed:
            if custom: assert checker._world_geometry_cache is None
            else:
                assert checker._world_geometry_cache is not None
                assert checker._world_geometry_cache.hits > 0
                assert checker._world_geometry_cache.misses > 0
    assert rows[0] == rows[1]


def test_newer_custom_producer_disables_cache_and_cancel_still_precedes_cached_verdict():
    f = Fixture(); checker = f.scene(f.node, Bin(), Tool(), np.ones((8, 3)))
    assert checker.sample(f.q, .017, False)
    cache = checker._world_geometry_cache; before = (cache.hits, cache.misses)
    original = f.node._collision_link_transforms
    f.node._collision_link_transforms = lambda *a, **k: original(*a, **k)
    q = f.q.copy(); q[1] = .01
    assert checker.sample(q, .017, False)
    assert (cache.hits, cache.misses) == before
    f.node._cancel.set()
    assert not checker.sample(q, .017, False)
    assert checker.last_rejection == {'reason': 'cancelled'}


@pytest.mark.parametrize('compound', [False, True])
def test_tool_owner_cache_and_compound_robot_grouping_preserve_exact_scene_trace(tmp_path, compound):
    parent = source_for_parent(tmp_path)
    rows = []
    for changed in (False, True):
        f = Fixture()
        if compound:
            original = f.node.carried_collision_meshes[0]
            handle = ModelLocalMesh(box(.05))
            extra = CollisionMesh(original.link, handle.snapshot()[0],
                                  np.array([box(.05).min(axis=(0, 1)),
                                            box(.05).max(axis=(0, 1))]), True, handle)
            # Preserve the same component order through concatenate and snapshot.
            f.node.carried_collision_meshes += (extra,)
        tool = Tool(); handle, surface = owned(3.)
        tool.local_surfaces = lambda aperture: {'tool': handle.snapshot()[0]}
        tool.model_local_surface = lambda value: handle if handle.matches(value) else None
        cls = f.scene if changed else actual_class(parent, f, scene=True)
        checker = cls(f.node, Bin(), tool, np.ones((8, 3)))
        outcomes = []
        for value in (0., .01, .02):
            q = f.q.copy(); q[1] = value
            outcomes.append((checker.sample(q, .017, False), checker.last_rejection,
                             checker.minimum_moving_left_z, checker.samples))
        rows.append((outcomes, f.events, f.node._self_collision_cache,
                     f.node._static_self_collision_cache, f.world_calls))
        if changed:
            cache = checker._world_geometry_cache
            assert cache.hits >= 2 * (len(f.node.carried_collision_meshes) + 1)
            assert any(key[0] is handle for key in cache._entries)
    assert rows[0] == rows[1]


@pytest.mark.parametrize('which', ['left', 'tool'])
@pytest.mark.parametrize('low', [.02, np.nextafter(.02, -np.inf), np.nextafter(.02, np.inf)])
def test_actual_scene_ground_boundary_and_minimum_are_exact(tmp_path, which, low):
    parent = source_for_parent(tmp_path)
    rows = []
    for changed in (False, True):
        f = Fixture(); tool = Tool()
        surface = box()
        surface[..., 2] = np.where(surface[..., 2] < 1., low, low + .2)
        handle = ModelLocalMesh(surface)
        if which == 'left':
            meshes = list(f.node.carried_collision_meshes)
            old = meshes[-1]
            meshes[-1] = CollisionMesh(old.link, handle.snapshot()[0],
                                      np.array([surface.min(axis=(0, 1)), surface.max(axis=(0, 1))]),
                                      True, handle)
            f.node.carried_collision_meshes = tuple(meshes)
        else:
            tool.local_surfaces = lambda aperture: {'tool': handle.snapshot()[0]}
            tool.model_local_surface = lambda value: handle if handle.matches(value) else None
        cls = f.scene if changed else actual_class(parent, f, scene=True)
        checker = cls(f.node, Bin(), tool, np.ones((8, 3)))
        result = checker.sample(f.q, .017, False)
        rows.append((result, checker.last_rejection, checker.minimum_moving_left_z, f.events))
        assert result is bool(low >= .02)
    assert rows[0] == rows[1]


def test_changed_immutable_model_invalidates_geometry_without_changing_body_policy(tmp_path):
    parent = source_for_parent(tmp_path)
    rows = []
    for changed in (False, True):
        f = Fixture(); cls = f.scene if changed else actual_class(parent, f, scene=True)
        checker = cls(f.node, Bin(), Tool(), np.ones((8, 3)))
        assert checker.sample(f.q, .017, False)
        old = f.node.carried_collision_meshes[-1]
        surface = old.triangles.copy(); surface[..., 2] += .3
        handle = ModelLocalMesh(surface)
        new = CollisionMesh(old.link, handle.snapshot()[0],
                            np.array([surface.min(axis=(0, 1)), surface.max(axis=(0, 1))]),
                            True, handle)
        f.node.carried_collision_meshes = (*f.node.carried_collision_meshes[:-1], new)
        # Existing body verdict cache policy is unchanged; use a new exact state.
        q = f.q.copy(); q[1] = .2
        assert checker.sample(q, .017, False)
        rows.append((f.events, checker.minimum_moving_left_z,
                     f.node._self_collision_cache, f.node._static_self_collision_cache))
        if changed:
            assert any(key[0] is handle for key in checker._world_geometry_cache._entries)
    assert rows[0] == rows[1]


def test_unchanged_scene_scope_preserves_original_world_cache_assertions():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    # The snapshot helper now also supports lazy cradle capture; its current
    # behavior is covered directly in test_cradle_sample_reuse.py.
    records = json.loads(FIXTURE.read_text())
    for rel in ('erc_phase1_solution/scene_checked_place.py',):
        record = records[rel]
        data = (ROOT / rel).read_bytes()
        from geometry_predicate_test_support import restore_predicate_bytes
        assert hashlib.sha256(restore_predicate_bytes(data, Path(rel).name)).hexdigest() == record['candidate_sha256']
        restored = restore_world_geometry_source(data.decode(), Path(rel).name)
        ast.parse(restored)
        with pytest.raises(AssertionError):
            restore_world_geometry_source(data.decode() + '\n# unknown edit\n', Path(rel).name)
