"""Bounded differentials against whole pinned parent modules; no ROS/assets.

Spy cases compare exact predicate inputs/order, not simulated physical safety.
The separate small-box cases exercise the actual inherited SAT predicate.
"""
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import types

import numpy as np
import pytest

from erc_phase1_solution import table_scene
from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
from erc_phase1_solution.geometry_array_checks import any_boolean, ExactRigidValidation
from geometry_predicate_test_support import (
    FIXTURE, FIXTURE_SHA256, restore_predicate_source, restore_predicate_bytes,
)

ROOT = Path(__file__).resolve().parents[1]


def parent_table():
    path = ROOT/'erc_phase1_solution/table_scene.py'
    text = restore_predicate_source(path.read_text(), path.name)
    namespace = {'__name__': 'erc_phase1_solution._pinned_predicate_parent',
                 '__package__': 'erc_phase1_solution'}
    exec(compile(text, str(path)+'::whole-parent', 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def scene():
    return dict(valid=True, model='erc_table_two_edge_rgbd_v1', frame='base_footprint',
                origin=[0., 0., 1.], rotation=np.eye(3).tolist(),
                modeled_margin_m=.005, registration_margin_m=.006,
                solids=table_scene._solid_bounds(), mesh_sha256=table_scene.TABLE_MESH_SHA256)


def triangles(centre=(0., 0., 0.), half=.03):
    vertices = np.asarray(list(itertools.product((-half, half), repeat=3)))+centre
    faces = np.asarray([[0,1,3],[0,3,2],[4,6,7],[4,7,5],
                        [0,4,5],[0,5,1],[2,3,7],[2,7,6],
                        [0,2,6],[0,6,4],[1,5,7],[1,7,3]])
    return np.array(vertices[faces], dtype=np.float64, order='C', copy=True)


def signature(value):
    return value.shape, value.dtype.str, value.tobytes(order='C')


def outcome(call):
    try:
        result = call()
        return 'return', signature(result) if isinstance(result, np.ndarray) else result
    except Exception as error:
        return 'raise', type(error).__name__, str(error)


@pytest.mark.parametrize('mask', [
    np.asarray(False), np.asarray(True), np.zeros(0, dtype=bool),
    np.zeros(3, dtype=bool), np.ones(3, dtype=bool), np.asarray([False,True,False]),
    np.eye(3, dtype=bool), np.eye(4, dtype=bool)[:, ::2],
])
def test_exact_boolean_reductions_preserve_value_type_and_skip_numpy_dispatch(mask, monkeypatch):
    original = np.any
    expected = original(mask)
    monkeypatch.setattr(np, 'any', lambda *_: pytest.fail('exact boolean used NumPy dispatch'))
    actual = any_boolean(mask)
    assert type(actual) is type(expected) and actual == expected


@pytest.mark.parametrize('value', [np.asarray([0., np.nan]), np.asarray([0,2]),
                                  [False,True], (0,0)])
def test_non_boolean_or_nonarray_fallback_preserves_original_dispatch(value, monkeypatch):
    original = np.any
    calls = []
    def observed(argument):
        calls.append(argument)
        return original(argument)
    monkeypatch.setattr(np, 'any', observed)
    expected = original(value)
    actual = any_boolean(value)
    assert len(calls) == 1 and calls[0] is value
    assert type(actual) is type(expected) and actual == expected


def test_ndarray_subclass_custom_array_function_is_not_bypassed():
    class Custom(np.ndarray):
        def __array_function__(self, function, types_, args, kwargs):
            assert function is np.any
            return 'custom-any'
    value = np.asarray([False,True]).view(Custom)
    assert any_boolean(value) == np.any(value) == 'custom-any'


def test_masked_array_dispatch_retained():
    value = np.ma.array([False, True], mask=[False, True])
    assert outcome(lambda: any_boolean(value)) == outcome(lambda: np.any(value))


@pytest.mark.parametrize('variant', ['nan','inf','bottom','rotation','shape'])
def test_bad_transform_errors_repeat_and_never_populate_success_cache(variant):
    value = np.eye(4)
    if variant == 'nan': value[0,3] = np.nan
    elif variant == 'inf': value[0,3] = np.inf
    elif variant == 'bottom': value[3,0] = .1
    elif variant == 'rotation': value[0,0] = 1.1
    else: value = np.eye(3)
    cache = ExactRigidValidation(table_scene._rigid)
    expected = outcome(lambda: table_scene._rigid(value))
    assert expected[0] == 'raise'
    for _ in range(2):
        assert outcome(lambda: cache.validate(value, table_scene._rigid)) == expected
    assert not cache._entries and cache.hits == 0


@pytest.mark.parametrize('variant', ['float32','view','fortran','subclass','list','endian'])
def test_unsupported_transform_forms_use_original_input_each_time(variant):
    value = np.eye(4)
    if variant == 'float32': value = value.astype(np.float32)
    elif variant == 'view': value = value.view()
    elif variant == 'fortran': value = np.asfortranarray(value)
    elif variant == 'subclass': value = value.view(type('Custom', (np.ndarray,), {}))
    elif variant == 'list': value = value.tolist()
    else: value = value.astype(np.dtype('>f8'))
    calls = []
    def validator(argument):
        calls.append(argument)
        return table_scene._rigid(argument)
    cache = ExactRigidValidation(validator)
    for _ in range(2):
        assert outcome(lambda: cache.validate(value, validator)) == outcome(lambda: table_scene._rigid(value))
    assert len(calls) == 2 and all(argument is value for argument in calls)
    assert cache.fallbacks == 2 and not cache._entries


def test_exact_transform_hit_fresh_metadata_and_one_ulp_and_signed_zero_misses():
    cache = ExactRigidValidation(table_scene._rigid)
    original = np.eye(4)
    first = cache.validate(original, table_scene._rigid)
    first.shape = (16,)
    second = cache.validate(original, table_scene._rigid)
    assert second.shape == (4,4) and not second.flags.writeable
    with pytest.raises(ValueError): second[0,3] = 5.
    negative_zero = original.copy(); negative_zero[0,3] = -0.
    one_ulp = original.copy(); one_ulp[0,3] = np.nextafter(0., 1.)
    for value in (negative_zero, one_ulp):
        assert signature(cache.validate(value, table_scene._rigid)) == signature(table_scene._rigid(value))
    assert cache.hits == 1 and cache.misses == 3 and len(cache._entries) == 3
    original[0,3] = 3.
    assert cache.validate(original, table_scene._rigid)[0,3] == 3.
    assert second[0,3] == 0.


def test_transform_cache_lru_is_bounded_and_evicted_inputs_are_revalidated():
    cache = ExactRigidValidation(table_scene._rigid)
    for offset in range(65):
        value = np.eye(4); value[0,3] = float(offset)
        cache.validate(value, table_scene._rigid)
    assert len(cache._entries) == 64 and sum(map(len, cache._entries)) == 8192
    assert np.eye(4).tobytes() not in cache._entries
    cache.validate(np.eye(4), table_scene._rigid)
    assert cache.misses == 66 and len(cache._entries) == 64


def test_changed_validator_falls_through_even_for_warm_input():
    cache = ExactRigidValidation(table_scene._rigid)
    value = np.eye(4)
    cache.validate(value, table_scene._rigid)
    calls = []
    def changed(argument):
        calls.append(argument)
        raise RuntimeError('changed validator')
    for _ in range(2):
        with pytest.raises(RuntimeError, match='changed validator'):
            cache.validate(value, changed)
    assert calls[0] is value and calls[1] is value and cache.hits == 0


def test_validator_changed_output_is_returned_without_success_entry():
    def changed(value):
        result = table_scene._rigid(value)
        result[0,3] = 1.
        return result
    cache = ExactRigidValidation(changed)
    assert cache.validate(np.eye(4), changed)[0,3] == 1.
    assert not cache._entries


@pytest.mark.parametrize('model', [False,True])
@pytest.mark.parametrize('shift', [0., .6, 4.])
@pytest.mark.parametrize('verdict', [False,True])
def test_table_parent_exact_predicate_arguments_order_and_result(model, shift, verdict, monkeypatch):
    from erc_phase1_solution import kinematics
    calls = []
    def predicate(corners, world, *, closed_surface):
        calls.append((signature(corners), signature(world), closed_surface))
        return verdict
    monkeypatch.setattr(kinematics, 'oriented_box_intersects_triangles', predicate)
    surface = triangles()
    if model: surface = ModelLocalMesh(surface)
    transform = np.eye(4)
    angle = .61
    transform[:2,:2] = [[math.cos(angle),-math.sin(angle)], [math.sin(angle),math.cos(angle)]]
    transform[:3,3] = [shift,0.,1.]
    rows = []
    for cls in (parent_table().TableSceneObstacle, table_scene.TableSceneObstacle):
        obj = cls(scene())
        sequence = []
        for _ in range(2):
            calls.clear()
            sequence.append((obj.intersects(surface, transform, True), obj.last_intersection, list(calls)))
        rows.append(sequence)
    assert rows[0] == rows[1]


@pytest.mark.parametrize('model', [False,True])
@pytest.mark.parametrize('x', [.7+.011+.03-1e-9, .7+.011+.03, .7+.011+.03+1e-9, 4.])
def test_actual_small_box_sat_boundary_matches_parent(model, x):
    surface = triangles()
    if model: surface = ModelLocalMesh(surface)
    transform = np.eye(4); transform[:3,3] = [x,0.,1.]
    old = parent_table().TableSceneObstacle(scene())
    new = table_scene.TableSceneObstacle(scene())
    assert (new.intersects(surface, transform, True), new.last_intersection) == (
            old.intersects(surface, transform, True), old.last_intersection)


def test_immutable_handle_skips_only_facet_scan_and_raw_mutation_still_rejects(monkeypatch):
    obj = table_scene.TableSceneObstacle(scene())
    raw = triangles()
    model = ModelLocalMesh(raw)
    transform = np.eye(4); transform[0,3] = 4.
    # Warm transform validation first, then detect only the large facet scan.
    obj.intersects(model, transform, True)
    original = np.isfinite
    visited = []
    def observed(value, *args, **kwargs):
        visited.append(np.asarray(value).shape)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(np, 'isfinite', observed)
    assert not obj.intersects(model, transform, True)
    assert raw.shape not in visited
    raw[0,0,0] = np.nan
    with pytest.raises(ValueError, match='invalid table collision surface'):
        obj.intersects(raw, transform, True)
    assert raw.shape in visited
    assert not obj.intersects(model, transform, True)


def test_handle_owner_mismatch_conservatively_returns_to_full_facet_validation(monkeypatch):
    obj = table_scene.TableSceneObstacle(scene())
    model = ModelLocalMesh(triangles())
    monkeypatch.setattr(ModelLocalMesh, 'matches', lambda *args: False)
    original = np.isfinite
    visited = []
    def observed(value, *args, **kwargs):
        visited.append(np.asarray(value).shape)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(np, 'isfinite', observed)
    obj.intersects(model, np.eye(4), False)
    assert model.snapshot()[0].shape in visited


@pytest.mark.parametrize('invalid_surface', [False,True])
def test_surface_then_transform_error_order_preserved(invalid_surface):
    class BadTransform:
        def __array__(self, *args, **kwargs):
            raise RuntimeError('transform evaluated')
    value = np.zeros((3,3)) if invalid_surface else triangles()
    old = parent_table().TableSceneObstacle(scene())
    new = table_scene.TableSceneObstacle(scene())
    assert outcome(lambda: new.intersects(value, BadTransform(), False)) == outcome(
        lambda: old.intersects(value, BadTransform(), False))
    assert not new._local_bounds._entries and not new._rigid_validation._entries


def test_table_subclass_and_legacy_new_instances_keep_uncached_transform_path():
    class Custom(table_scene.TableSceneObstacle): pass
    custom = Custom(scene())
    assert custom._rigid_validation is None
    legacy = table_scene.TableSceneObstacle(scene())
    del legacy._rigid_validation
    assert outcome(lambda: custom.intersects(triangles(), np.eye(4), False)) == outcome(
        lambda: legacy.intersects(triangles(), np.eye(4), False))


@pytest.mark.parametrize('filename', ['table_scene.py','scene_checked_place.py'])
def test_complete_parent_inverse_and_unrelated_change_rejection(filename):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    data = (ROOT/'erc_phase1_solution'/filename).read_bytes()
    restored = restore_predicate_bytes(data, filename)
    record = json.loads(FIXTURE.read_bytes())['erc_phase1_solution/'+filename]
    assert hashlib.sha256(restored).hexdigest() == record['parent_sha256']
    ast.parse(restored)
    with pytest.raises(AssertionError):
        restore_predicate_source(data.decode()+'\n# undeclared\n', filename)


def test_scene_only_import_and_boolean_reduction_sites_change():
    path = ROOT/'erc_phase1_solution/scene_checked_place.py'
    old = ast.parse(restore_predicate_source(path.read_text(), path.name))
    new = ast.parse(path.read_text())
    old_class = next(node for node in old.body if isinstance(node, ast.ClassDef) and node.name == 'PlaceSceneChecker')
    new_class = next(node for node in new.body if isinstance(node, ast.ClassDef) and node.name == 'PlaceSceneChecker')
    before = {node.name: node for node in old_class.body if isinstance(node, ast.FunctionDef)}
    after = {node.name: node for node in new_class.body if isinstance(node, ast.FunctionDef)}
    assert before.keys() == after.keys()
    for name in before.keys()-{'sample'}:
        assert ast.dump(before[name], include_attributes=False) == ast.dump(after[name], include_attributes=False)
    sample = after['sample']
    assert isinstance(sample.body[0], ast.ImportFrom)
    sample.body.pop(0)
    changed = 0
    for node in ast.walk(sample):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'any_boolean':
            node.func = ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='any', ctx=ast.Load())
            changed += 1
    assert changed == 4
    assert ast.dump(before['sample'], include_attributes=False) == ast.dump(sample, include_attributes=False)
