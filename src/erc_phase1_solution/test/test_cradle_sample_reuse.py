"""Current cradle behavior and lazy cache ordering; no mirrored source fixtures."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution import shelf_cradle_geometry as cradle
from erc_phase1_solution import sample_collision_snapshot as snapshots
from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
from erc_phase1_solution.kinematics import CollisionMesh
from test_sample_collision_snapshot import Fixture, box, signature


def fixture(*, rejection=False, immutable=True):
    f = Fixture(rejection=rejection, immutable=immutable)
    n = f.node
    n.chain.active_names = tuple('q'+str(i) for i in range(8))
    n.chain.lower = np.full(8, -2.)
    n.chain.upper = np.full(8, 2.)
    n.carried_book_dimensions = np.array([.16, .02, .25])
    n.carried_transition_samples = 3
    n.carried_shelf_margin = .02
    n._shelf_cradle_geometry = NS(
        local_surfaces=lambda aperture, fingers: {cradle.PALM_COLLISION_LINK: box(20.)},
        watertight={cradle.PALM_COLLISION_LINK: True})
    # The existing actual-method fixture uses class-bound ordinary producers.
    # Support omitted parked context, as the real producer does, for fallback tests.
    def world(q, **context):
        f.world_calls += 1
        return f.robot(q, context.get('right_positions', f.right),
                       context.get('head_positions', f.head))
    f.world = world
    return f


def sweep(f, monkeypatch, *, lazy=True, moving=False, omitted=None):
    context = dict(right_positions=f.right, head_positions=f.head)
    if omitted is not None: context.pop(omitted)
    end = f.q.copy()
    if moving: end[0] += .04
    with monkeypatch.context() as patch:
        if not lazy:
            patch.setattr(snapshots, '_capture_cradle_robot_snapshot', lambda *a: None)
        return cradle.check_cradle_tool_sweep(f.node, [5., 0., 1.], f.q,
            f.q, end, None, aperture=.02, **context)


@pytest.mark.parametrize('moving', [False, True])
@pytest.mark.parametrize('rejection', [False, True])
def test_current_sweep_preserves_order_first_verdict_and_caches(monkeypatch, moving, rejection):
    old = fixture(rejection=rejection)
    new = fixture(rejection=rejection)
    expected = sweep(old, monkeypatch, lazy=False, moving=moving)
    assert sweep(new, monkeypatch, moving=moving) == expected
    assert old.events == new.events
    assert old.node._self_collision_cache == new.node._self_collision_cache
    assert old.node._static_self_collision_cache == new.node._static_self_collision_cache
    if rejection:
        assert expected.startswith('cradle_robot_self_intersection:')
        assert old.world_calls == new.world_calls == 1
    else:
        assert expected is None
        count = 3 if moving else 1
        assert old.world_calls == 2 * count and new.world_calls == count


@pytest.mark.parametrize('rejection', [False, True])
def test_cache_hit_keeps_original_world_call_count(monkeypatch, rejection):
    f = fixture(rejection=rejection)
    expected = sweep(f, monkeypatch)
    f.world_calls = 0
    events = list(f.events)
    assert sweep(f, monkeypatch) == expected
    assert f.events == events
    assert f.world_calls == (0 if rejection else 1)


def test_cached_rejection_cannot_be_masked_by_future_producer_failure(monkeypatch):
    f = fixture(rejection=True)
    expected = sweep(f, monkeypatch)
    def broken(*args, **kwargs):
        raise RuntimeError('producer must not run after a cached rejection')
    f.world = broken
    assert sweep(f, monkeypatch) == expected


def test_body_miss_preserves_producer_failure_reason(monkeypatch):
    results = []
    for lazy in (False, True):
        f = fixture()
        def broken(*args, **kwargs): raise RuntimeError('producer failed')
        f.world = broken
        results.append(sweep(f, monkeypatch, lazy=lazy))
    assert results == ['cradle_tool_geometry_unavailable:RuntimeError:producer failed'] * 2


@pytest.mark.parametrize('omitted', ['right_positions', 'head_positions'])
def test_missing_explicit_context_keeps_original_path(monkeypatch, omitted):
    f = fixture()
    assert sweep(f, monkeypatch, omitted=omitted) is None
    assert f.world_calls == 2


@pytest.mark.parametrize('custom', ['consumer', 'producer', 'mutable_mesh'])
def test_custom_family_keeps_original_signature_and_work(monkeypatch, custom):
    f = fixture(immutable=custom != 'mutable_mesh')
    if custom == 'consumer':
        original = f.node._robot_self_collision
        def body(q, *, right_positions, head_positions):
            return original(q, right_positions=right_positions, head_positions=head_positions)
        f.node._robot_self_collision = body
    elif custom == 'producer':
        original = f.node._world_collision_surfaces
        f.node._world_collision_surfaces = lambda *args, **kwargs: original(*args, **kwargs)
    assert sweep(f, monkeypatch) is None
    assert f.world_calls == 2


def test_cancellation_precedes_snapshot_and_world(monkeypatch):
    f = fixture()
    f.node._cancel.set()
    def forbidden(*args): raise AssertionError('cancelled sweep generated a snapshot')
    monkeypatch.setattr(snapshots, '_capture_cradle_robot_snapshot', forbidden)
    assert sweep(f, monkeypatch) == 'cradle_tool_cancelled'
    assert f.world_calls == 0 and f.events == []


@pytest.mark.parametrize('field', ['q', 'right', 'head'])
def test_exact_state_change_rejects_before_lazy_materialization(field):
    f = fixture()
    snapshot = snapshots._capture_cradle_robot_snapshot(f.node, f.q, f.right, f.head)
    assert snapshot is not None and f.world_calls == 0
    getattr(f, field)[0] = np.nextafter(0., 1.)
    assert snapshot.resolve(f.node, f.q, f.right, f.head) is None
    assert f.world_calls == 0


def test_model_replacement_rejects_before_lazy_materialization():
    f = fixture()
    snapshot = snapshots._capture_cradle_robot_snapshot(f.node, f.q, f.right, f.head)
    mesh = f.node.carried_collision_meshes[0]
    local = ModelLocalMesh(box(.08))
    f.node.carried_collision_meshes = (CollisionMesh(mesh.link, local.snapshot()[0],
        mesh.bounds, True, local), *f.node.carried_collision_meshes[1:])
    assert snapshot.resolve(f.node, f.q, f.right, f.head) is None
    assert f.world_calls == 0


def test_state_is_rechecked_after_producer_returns():
    f = fixture()
    snapshot = snapshots._capture_cradle_robot_snapshot(f.node, f.q, f.right, f.head)
    original = f.world
    def changed(q, **context):
        result = original(q, **context)
        q[0] += .01
        return result
    f.world = changed
    assert snapshot.resolve(f.node, f.q, f.right, f.head) is None
    assert snapshot.resolved(f.node, f.q, f.right, f.head) is None
    assert f.world_calls == 1


def test_lazy_capture_keeps_exact_facets_bounds_and_immutable_views():
    f = fixture()
    expected = f.robot()
    snapshot = snapshots._capture_cradle_robot_snapshot(f.node, f.q, f.right, f.head)
    surfaces, bounds = snapshot.resolve(f.node, f.q, f.right, f.head)
    assert f.world_calls == 1 and tuple(surfaces) == tuple(expected)
    for link, original in expected.items():
        assert signature(surfaces[link]) == signature(original)
        assert signature(bounds[link]) == signature(np.asarray([
            original.min(axis=(0, 1)), original.max(axis=(0, 1))]))
        assert not surfaces[link].flags.writeable and not bounds[link].flags.writeable
    snapshot.resolve(f.node, f.q, f.right, f.head)
    assert f.world_calls == 1
