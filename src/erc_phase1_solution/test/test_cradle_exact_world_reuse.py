"""Exact reuse differentials against hash-pinned original sweep/snapshot source."""
import importlib.util
import hashlib
from pathlib import Path
import threading

import numpy as np
import pytest

from erc_phase1_solution import shelf_cradle_geometry as current
from erc_phase1_solution import sample_collision_snapshot as snapshots
from erc_phase1_solution.exact_world_geometry import ExactWorldGeometry
from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
from erc_phase1_solution.kinematics import CollisionMesh
from erc_phase1_solution.pure_geometry_owner import RobotCollisionGeometry, RIGHT_ARM_JOINTS, HEAD_JOINTS
from test_cradle_sample_reuse import fixture
from test_sample_collision_snapshot import box, signature


BASELINE_SHA256 = {
    'sample_collision_snapshot': '8d2e6cda3f68242ebe61348dcb3967f1df1f74fb6839912061bb56a02aa83624',
    'shelf_cradle_geometry': 'b857bf5c97ce34c759f7228e13327d69d13c3b4a64fe469c4c02e4141f87d295',
}


def baseline(name):
    path = Path(__file__).parent / 'fixtures' / ('cradle_exact_reuse_original_' + name + '.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BASELINE_SHA256[name]
    spec = importlib.util.spec_from_file_location('erc_phase1_solution._baseline_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old_cradle = baseline('shelf_cradle_geometry')
old_snapshots = baseline('sample_collision_snapshot')


def ordinary_fixture(**kwargs):
    f = fixture(**kwargs)
    n = f.node
    for name in ('_resolved_right_positions', '_resolved_head_positions',
                 '_right_joint_positions', '_head_joint_positions'):
        if name in vars(n): delattr(n, name)
        setattr(type(n), name, getattr(RobotCollisionGeometry, name))
    n.joints = dict(zip(RIGHT_ARM_JOINTS, f.right))
    n.joints.update(zip(HEAD_JOINTS, f.head))
    n._lock = threading.Lock()
    f.transform_calls = 0
    original = f.transforms
    def transforms(q, **context):
        f.transform_calls += 1
        return original(q, **context)
    f.transforms = transforms
    assert current._ordinary_cradle_state_access(n)
    return f


def run(f, monkeypatch, *, version, moving=True, implicit=()):
    context = dict(right_positions=f.right, head_positions=f.head)
    for name in implicit: context.pop(name)
    end = f.q.copy()
    if moving: end[0] += .04
    module = old_cradle if version == 'old' else current
    with monkeypatch.context() as patch:
        if version == 'old':
            patch.setattr(snapshots, '_capture_cradle_robot_snapshot',
                          old_snapshots._capture_cradle_robot_snapshot)
        return module.check_cradle_tool_sweep(f.node, [5., 0., 1.], f.q,
                    f.q, end, None, aperture=.02, **context)


@pytest.mark.parametrize('rejection', [False, True])
@pytest.mark.parametrize('moving', [False, True])
@pytest.mark.parametrize('implicit', [(), ('right_positions',), ('head_positions',),
                                     ('right_positions', 'head_positions')])
def test_exact_sweep_verdict_order_and_body_caches(monkeypatch, rejection, moving, implicit):
    old, new = ordinary_fixture(rejection=rejection), ordinary_fixture(rejection=rejection)
    kwargs = dict(moving=moving, implicit=implicit)
    before = run(old, monkeypatch, version='old', **kwargs)
    after = run(new, monkeypatch, version='new', **kwargs)
    assert before == after
    assert old.events == new.events
    assert old.node._self_collision_cache == new.node._self_collision_cache
    assert old.node._static_self_collision_cache == new.node._static_self_collision_cache
    old_count, new_count = old.world_calls, new.world_calls
    if implicit and not rejection:
        assert old_count == 2 * (3 if moving else 1)
    assert new_count == 0  # original transforms and grouping run in the exact cache helper
    assert new.transform_calls == (1 if rejection else 3 if moving else 1)
    before_events, after_events = list(old.events), list(new.events)
    assert run(old, monkeypatch, version='old', **kwargs) == before
    assert run(new, monkeypatch, version='new', **kwargs) == after
    assert old.events == before_events and new.events == after_events
    if rejection:
        assert old.world_calls == old_count and new.world_calls == new_count


def test_each_sample_captures_both_implicit_groups_under_one_lock(monkeypatch):
    f = ordinary_fixture()
    class UpdatingLock:
        entries = 0
        def __enter__(self):
            self.entries += 1
            f.node.joints[RIGHT_ARM_JOINTS[0]] = self.entries * 1e-3
            f.node.joints[HEAD_JOINTS[0]] = -self.entries * 2e-3
        def __exit__(self, *unused): return False
    f.node._lock = lock = UpdatingLock()
    seen = []
    original = f.transforms
    def recorded(q, **context):
        seen.append((context['right_positions'].copy(), context['head_positions'].copy()))
        return original(q, **context)
    f.transforms = recorded
    assert run(f, monkeypatch, version='new', implicit=('right_positions', 'head_positions')) is None
    assert lock.entries == 3 and len(seen) == 3
    assert [(r[0], h[0]) for r, h in seen] == [(i*.001, -i*.002) for i in range(1, 4)]


@pytest.mark.parametrize('implicit', [('right_positions',), ('head_positions',)])
def test_explicit_group_retained_when_other_is_measured(implicit):
    f = ordinary_fixture()
    f.node.joints[RIGHT_ARM_JOINTS[0]] = .01
    f.node.joints[HEAD_JOINTS[0]] = -.02
    supplied = dict(right_positions=f.right, head_positions=f.head)
    supplied.pop(implicit[0])
    result = current._paired_cradle_context(f.node, supplied)
    assert result['right_positions'][0] == (.01 if implicit[0] == 'right_positions' else 0.)
    assert result['head_positions'][0] == (-.02 if implicit[0] == 'head_positions' else 0.)


@pytest.mark.parametrize('changed', ['_resolved_right_positions', '_resolved_head_positions',
                                   '_right_joint_positions', '_head_joint_positions',
                                   '_world_collision_surfaces', '_collision_link_transforms'])
def test_custom_method_rebind_falls_back(changed):
    f = ordinary_fixture()
    old = getattr(f.node, changed)
    setattr(f.node, changed, lambda *a, **kw: old(*a, **kw))
    assert not current._ordinary_cradle_state_access(f.node)


def test_method_rebind_mid_sweep_is_rechecked(monkeypatch):
    f = ordinary_fixture()
    original = f.transforms
    def changed(q, **context):
        result = original(q, **context)
        if f.transform_calls == 1:
            old = f.node._right_joint_positions
            f.node._right_joint_positions = lambda: old()
        return result
    f.transforms = changed
    assert run(f, monkeypatch, version='new', implicit=('right_positions', 'head_positions')) is None
    # First sample uses exact shared output; the remaining custom-reader
    # samples take original separate body/world paths.
    assert f.world_calls == 4


@pytest.mark.parametrize('compound', [False, True])
def test_cached_snapshot_facets_bounds_owners_and_exact_context(compound):
    f = ordinary_fixture()
    if compound:
        first = f.node.carried_collision_meshes[0]
        local = ModelLocalMesh(box(.08))
        extra = CollisionMesh(first.link, local.snapshot()[0], first.bounds, True, local)
        f.node.carried_collision_meshes = (first, extra, *f.node.carried_collision_meshes[1:])
    cache = ExactWorldGeometry()
    for step in range(3):
        if step == 2: f.head[0] = np.nextafter(0., 1.)
        snapshot = snapshots._capture_cradle_robot_snapshot(f.node, f.q, f.right, f.head, _world_cache=cache)
        expected = f.robot()
        actual, bounds = snapshot.resolve(f.node, f.q, f.right, f.head)
        assert tuple(actual) == tuple(expected)
        for key, value in expected.items():
            assert signature(actual[key]) == signature(value)
            assert signature(bounds[key]) == signature(np.asarray([value.min(axis=(0,1)), value.max(axis=(0,1))]))
            assert not actual[key].flags.writeable and not bounds[key].flags.writeable
    assert cache.hits > 0 and cache.misses == 2 * len(f.node.carried_collision_meshes)


def test_cached_body_rejection_precedes_cache_capture(monkeypatch):
    f = ordinary_fixture(rejection=True)
    expected = run(f, monkeypatch, version='new')
    def fail(*unused): raise AssertionError('cached rejection reached geometry cache')
    monkeypatch.setattr(ExactWorldGeometry, 'capture', fail)
    assert run(f, monkeypatch, version='new') == expected


def test_cancellation_precedes_paired_state_read(monkeypatch):
    f = ordinary_fixture()
    f.node._cancel.set()
    def fail(*unused): raise AssertionError('cancelled sample read parked state')
    monkeypatch.setattr(current, '_paired_cradle_context', fail)
    assert run(f, monkeypatch, version='new') == 'cradle_tool_cancelled'
    assert f.world_calls == f.transform_calls == 0


@pytest.mark.parametrize('case', ['shelf', 'floor', 'intersection', 'invalid_book', 'invalid_aperture', 'invalid_joint'])
def test_full_first_failure_reason_and_tool_narrow_arguments_match(monkeypatch, case):
    outputs = []
    for version in ('old', 'new'):
        f = ordinary_fixture()
        module = old_cradle if version == 'old' else current
        shelf, aperture = None, .02
        if case == 'shelf': shelf = .05
        if case in ('floor', 'intersection'):
            surface = box(0.)
            if case == 'floor': surface[:,:,2] -= 2.
            f.node._shelf_cradle_geometry.local_surfaces = lambda *a: {current.PALM_COLLISION_LINK: surface}
        if case == 'invalid_book': f.node.carried_book_dimensions[0] = -1.
        if case == 'invalid_aperture': aperture = -.01
        if case == 'invalid_joint': f.q[1] = 100.
        narrow_calls = []
        original = module.triangle_meshes_intersect
        def recorded(a, b, **flags):
            narrow_calls.append((signature(a.surface), signature(b.surface), flags))
            return original(a, b, **flags)
        with monkeypatch.context() as patch:
            patch.setattr(module, 'triangle_meshes_intersect', recorded)
            if version == 'old':
                patch.setattr(snapshots, '_capture_cradle_robot_snapshot', old_snapshots._capture_cradle_robot_snapshot)
            reason = module.check_cradle_tool_sweep(f.node, [5.,0.,1.], f.q, f.q, f.q,
                                                    shelf, aperture=aperture)
        outputs.append((reason, narrow_calls, f.events))
    assert outputs[0] == outputs[1]
    assert outputs[0][0] is not None
    if case == 'intersection':
        assert outputs[0][0].startswith('cradle_tool_robot_intersection:')
        assert outputs[0][1]
