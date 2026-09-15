"""Prepared immutable-model/raw-path tests; no ROS, simulator or official assets.

These tests are intentionally unexecuted while the physical trial owns CPU.
Obstacle predicates are controlled spies except the small URDF box loader.
"""
import ast
import copy
import itertools
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution.exact_local_bounds import ExactLocalBounds, ModelLocalMesh
from erc_phase1_solution.kinematics import CollisionMesh, load_urdf_collision_meshes

PACKAGE = Path(__file__).resolve().parents[1] / 'erc_phase1_solution'


def triangles():
    return np.array([[[-.2, -.2, -.2], [.2, -.2, .2], [0., .2, 0.]],
                     [[-.1, -.1, .1], [.1, .1, -.1], [0., 0., .2]]], dtype=np.float64)


def signature(value):
    array = np.asarray(value)
    return array.shape, array.dtype.str, array.tobytes()


def test_snapshot_and_extrema_are_exact_and_isolated_from_source_aliases():
    raw = triangles()
    raw[0, 0, 0] = -0.
    original = raw.copy()
    alias = raw.view()
    model = ModelLocalMesh(raw)
    values = model.snapshot()
    assert signature(values[0]) == signature(original)
    assert signature(values[1]) == signature(original.min(axis=(0, 1)))
    assert signature(values[2]) == signature(original.max(axis=(0, 1)))
    assert model.nbytes == original.nbytes + 48
    alias[:] = 90.
    assert signature(model.snapshot()[0]) == signature(original)
    assert not model.matches(raw)
    for value in values:
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(AttributeError):
        model._content = b'forbidden'
    with pytest.raises(AttributeError):
        del model._shape


def test_array_metadata_changes_cannot_corrupt_future_snapshot_or_bounds():
    model = ModelLocalMesh(triangles())
    before = [signature(value) for value in model.snapshot()]
    values = model.snapshot()
    values[0].shape = (18,)
    values[1].shape = (1, 3)
    values[2].dtype = np.uint8
    assert not model.matches(values[0])
    assert [signature(value) for value in model.snapshot()] == before
    assert model.matches(model.snapshot()[0])


@pytest.mark.parametrize('change', ['copy', 'readonly_raw', 'readonly_alias', 'subclass',
                                    'transpose', 'dtype', 'other_handle'])
def test_binding_accepts_only_complete_exact_immutable_byte_owner(change):
    raw = triangles()
    model = ModelLocalMesh(raw)
    surface = model.snapshot()[0]
    assert model.matches(surface)
    if change == 'copy': surface = surface.copy()
    elif change == 'readonly_raw': surface = raw; surface.setflags(write=False)
    elif change == 'readonly_alias': surface = raw.view(); surface.setflags(write=False)
    elif change == 'subclass':
        class Custom(np.ndarray): pass
        surface = surface.view(Custom)
    elif change == 'transpose': surface = surface.transpose(0, 2, 1)
    elif change == 'dtype': surface = surface.view(np.uint64)
    elif change == 'other_handle': surface = ModelLocalMesh(raw).snapshot()[0]
    assert not model.matches(surface)


@pytest.mark.parametrize('variant', ['view', 'subclass', 'float32', 'foreign_endian',
                                    'empty', 'shape', 'noncontiguous', 'nan', 'infinity', 'list'])
def test_model_constructor_rejects_unsupported_or_nonfinite_input(variant):
    raw = triangles()
    if variant == 'view': raw = raw.view()
    elif variant == 'subclass':
        class Custom(np.ndarray): pass
        raw = raw.view(Custom)
    elif variant == 'float32': raw = raw.astype(np.float32)
    elif variant == 'foreign_endian': raw = raw.astype('>f8')
    elif variant == 'empty': raw = np.empty((0, 3, 3))
    elif variant == 'shape': raw = np.zeros((3, 3))
    elif variant == 'noncontiguous': raw = raw[:, ::-1]
    elif variant == 'nan': raw[0, 0, 0] = math.nan
    elif variant == 'infinity': raw[0, 0, 0] = math.inf
    elif variant == 'list': raw = raw.tolist()
    with pytest.raises(ValueError):
        ModelLocalMesh(raw)


def test_model_capture_never_uses_content_dictionary_or_retains_an_extra_entry():
    model = ModelLocalMesh(triangles())
    cache = ExactLocalBounds(maximum_entries=1, maximum_bytes=1)
    class NoLookup(dict):
        def get(self, *args): raise AssertionError('content lookup used for model handle')
    cache._entries = NoLookup()
    first, second = cache.capture(model), cache.capture(model)
    assert [signature(x) for x in first] == [signature(x) for x in second]
    assert first[0] is not second[0]  # array metadata is never shared
    assert cache._entries == {} and cache._bytes == 0


def test_raw_readonly_array_flag_reset_and_mutation_keep_exact_content_invalidation():
    raw = triangles()
    cache = ExactLocalBounds()
    raw.setflags(write=False)
    first = cache.capture(raw)
    raw.setflags(write=True)
    raw[:] += 10.
    second = cache.capture(raw)
    assert first is not second
    assert signature(second[0]) == signature(raw)
    assert second[1][0] > first[2][0]


def urdf(tmp_path, count=2):
    path = tmp_path / 'boxes.urdf'
    collisions = ''.join(
        f'<collision><origin xyz="{i*.3} 0 0" rpy="0 0 .2"/>'
        '<geometry><box size=".1 .2 .3"/></geometry></collision>' for i in range(count))
    path.write_text('<robot name="fixture"><link name="body">' + collisions + '</link></robot>')
    return path


def test_loader_opt_in_preserves_exact_order_triangles_bounds_and_watertight(tmp_path):
    path = urdf(tmp_path)
    resolver = lambda _: pytest.fail('box loader must not resolve a mesh package')
    original = load_urdf_collision_meshes(path, ['body'], resolver)
    frozen = load_urdf_collision_meshes(path, ['body'], resolver, immutable_local=True)
    assert len(original) == len(frozen) == 2
    for a, b in zip(original, frozen):
        assert a.local_mesh is None and a.triangles.flags.writeable
        assert type(b.local_mesh) is ModelLocalMesh and b.local_mesh.matches(b.triangles)
        assert a.link == b.link and a.watertight == b.watertight
        assert signature(a.triangles) == signature(b.triangles)
        assert signature(a.bounds) == signature(b.bounds)
        with pytest.raises(ValueError): b.triangles.setflags(write=True)
    custom = CollisionMesh('custom', triangles(), np.zeros((2, 3)), False)
    assert custom.local_mesh is None and custom.triangles.flags.writeable


def test_loader_total_handle_entry_and_byte_budget_fall_back_to_raw(tmp_path, monkeypatch):
    path = urdf(tmp_path, 4)
    resolver = lambda _: None
    monkeypatch.setattr(ModelLocalMesh, 'MAXIMUM_ENTRIES', 2)
    meshes = load_urdf_collision_meshes(path, ['body'], resolver, immutable_local=True)
    assert [m.local_mesh is not None for m in meshes] == [True, True, False, False]
    assert all(m.triangles.flags.writeable for m in meshes[2:])
    monkeypatch.setattr(ModelLocalMesh, 'MAXIMUM_ENTRIES', 64)
    monkeypatch.setattr(ModelLocalMesh, 'MAXIMUM_BYTES', meshes[0].local_mesh.nbytes)
    meshes = load_urdf_collision_meshes(path, ['body'], resolver, immutable_local=True)
    assert [m.local_mesh is not None for m in meshes] == [True, False, False, False]


def obstacle_class(kind, predicate):
    filename = 'scene_checked_place.py' if kind == 'bin' else 'table_scene.py'
    classname = 'NominalBinObstacle' if kind == 'bin' else 'TableSceneObstacle'
    tree = ast.parse((PACKAGE / filename).read_text())
    nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in {classname, '_finite', '_rigid'}]
    scope = dict(np=np, math=math, itertools=itertools, ExactLocalBounds=ExactLocalBounds,
                 oriented_box_intersects_triangles=predicate)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, 'exec'), scope)
    return scope[classname]


def obstacle(kind, predicate, monkeypatch):
    if kind == 'table':
        from erc_phase1_solution import kinematics
        monkeypatch.setattr(kinematics, 'oriented_box_intersects_triangles', predicate)
    cls = obstacle_class(kind, predicate)
    result = cls.__new__(cls)
    result.rotation = np.eye(3)
    result.origin = np.zeros(3)
    result.bounds = np.asarray([[-.5]*3, [.5]*3])
    corners = np.asarray(list(itertools.product((-.5, .5), repeat=3)))
    result.material_bounds = [result.bounds]
    result.material_corners = [corners]
    result.solids = [('solid', result.bounds, corners)]
    result.last_intersection = None
    result._local_bounds = ExactLocalBounds()
    return result


@pytest.mark.parametrize('kind', ['bin', 'table'])
@pytest.mark.parametrize('shift', [0., .4, 5.])
@pytest.mark.parametrize('verdict', [False, True])
def test_handle_raw_obstacle_predicate_arguments_order_and_first_result_match(kind, shift, verdict, monkeypatch):
    calls = []
    def predicate(corners, world, *, closed_surface):
        calls.append((signature(corners), signature(world), closed_surface))
        return verdict
    obj = obstacle(kind, predicate, monkeypatch)
    raw = triangles()
    model = ModelLocalMesh(raw)
    transform = np.eye(4)
    angle = .61
    transform[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    transform[0, 3] = shift
    original = obj.intersects(raw, transform, True), obj.last_intersection
    expected = list(calls)
    calls.clear()
    changed = obj.intersects(model, transform, True), obj.last_intersection
    assert changed == original and calls == expected


@pytest.mark.parametrize('kind', ['bin', 'table'])
def test_handle_does_not_cache_transform_or_verdict_and_uses_its_own_snapshot(kind, monkeypatch):
    calls = []
    def predicate(corners, world, **kwargs):
        calls.append(signature(world))
        return True
    obj = obstacle(kind, predicate, monkeypatch)
    raw = triangles()
    model = ModelLocalMesh(raw)
    original = raw.copy()
    raw[:] += 10.
    near = np.eye(4)
    far = near.copy(); far[0, 3] = 10.
    assert obj.intersects(model, near, False)
    assert calls == [signature(original)]
    assert not obj.intersects(model, far, False)
    assert not obj.intersects(raw, near, False)
    del obj._local_bounds
    assert obj.intersects(model, near, False)


def test_table_handle_keeps_transform_validation_before_capture(monkeypatch):
    obj = obstacle('table', lambda *a, **k: False, monkeypatch)
    obj._local_bounds.capture = lambda *_: pytest.fail('cache visited before bad transform')
    class BadTransform:
        def __array__(self, *args, **kwargs): raise RuntimeError('transform visited')
    with pytest.raises(RuntimeError, match='transform visited'):
        obj.intersects(ModelLocalMesh(triangles()), BadTransform(), False)


@pytest.mark.parametrize('stale_binding', [False, True])
def test_scene_custom_obstacle_keeps_raw_array_signature_and_compound_order(stale_binding):
    import test_scene_world_bounds_differential as shared
    fixture = shared.Fixture(compound=True, screen=False)
    for mesh in fixture.meshes:
        mesh.local_mesh = ModelLocalMesh(mesh.triangles)
        mesh.triangles = mesh.local_mesh.snapshot()[0]
        if stale_binding:
            mesh.triangles = mesh.triangles.copy() + .001
    scope = shared.classes(PACKAGE / 'scene_checked_place.py', fixture)
    # Both exact built-in types differ from the fixture's custom observers.
    scope.update(NominalBinObstacle=type('BuiltInBin', (), {}),
                 TableSceneObstacle=type('BuiltInTable', (), {}))
    checker = scope['PlaceSceneChecker'](fixture.node, fixture.obstacle, fixture.tool,
                                         fixture.attached, fixture.table, None)
    seen = []
    def check(surface, transform, watertight):
        assert type(surface) is np.ndarray
        seen.append(signature(surface))
        return False
    fixture.obstacle.intersects = check
    fixture.table.intersects = check
    assert checker.sample(shared.Q, .017, True)
    expected = [signature(mesh.triangles) for mesh in fixture.meshes for _ in range(2)]
    assert seen[:4] == expected


def test_scene_exact_builtin_obstacles_receive_model_handles_but_tools_stay_raw(monkeypatch):
    import test_scene_world_bounds_differential as shared
    fixture = shared.Fixture(compound=True, screen=False)
    fixture.obstacle = obstacle('bin', lambda *a, **k: False, monkeypatch)
    fixture.table = obstacle('table', lambda *a, **k: False, monkeypatch)
    fixture.obstacle.book_intersects = lambda *_: False
    fixture.table.intersects_box = lambda *_: False
    for mesh in fixture.meshes:
        mesh.local_mesh = ModelLocalMesh(mesh.triangles)
        mesh.triangles = mesh.local_mesh.snapshot()[0]
    seen = []
    for obj in [fixture.obstacle, fixture.table]:
        original = obj.intersects
        def observe(surface, transform, watertight, operation=original):
            seen.append(surface)
            return operation(surface, transform, watertight)
        obj.intersects = observe
    scope = shared.classes(PACKAGE / 'scene_checked_place.py', fixture)
    scope.update(NominalBinObstacle=type(fixture.obstacle), TableSceneObstacle=type(fixture.table))
    checker = scope['PlaceSceneChecker'](fixture.node, fixture.obstacle, fixture.tool,
                                         fixture.attached, fixture.table, None)
    assert checker.sample(shared.Q, .017, True)
    assert all(type(value) is ModelLocalMesh for value in seen[:4])
    assert all(type(value) is np.ndarray for value in seen[4:])
    assert seen[0] is seen[1] is fixture.meshes[0].local_mesh
    assert seen[2] is seen[3] is fixture.meshes[1].local_mesh


def test_stale_handle_cannot_hide_new_raw_geometry_collision_in_builtin_bin(monkeypatch):
    import test_scene_world_bounds_differential as shared
    fixture = shared.Fixture(screen=False)
    fixture.obstacle = obstacle('bin', lambda *a, **k: True, monkeypatch)
    fixture.table = None
    fixture.obstacle.book_intersects = lambda *_: False
    mesh = fixture.meshes[0]
    mesh.local_mesh = ModelLocalMesh(mesh.triangles)  # Old box is above the bin.
    mesh.triangles = mesh.local_mesh.snapshot()[0].copy() - [0., 0., 1.]
    assert not mesh.local_mesh.matches(mesh.triangles)
    seen = []
    original = fixture.obstacle.intersects
    def observe(surface, transform, watertight):
        seen.append(surface)
        return original(surface, transform, watertight)
    fixture.obstacle.intersects = observe
    scope = shared.classes(PACKAGE / 'scene_checked_place.py', fixture)
    scope.update(NominalBinObstacle=type(fixture.obstacle), TableSceneObstacle=type('Table', (), {}))
    checker = scope['PlaceSceneChecker'](fixture.node, fixture.obstacle, fixture.tool,
                                         fixture.attached, None, None)
    assert not checker.sample(shared.Q, .017, True)
    assert checker.last_rejection == {'reason': 'robot_bin', 'link': mesh.link}
    assert seen == [mesh.triangles]
    assert type(seen[0]) is np.ndarray
