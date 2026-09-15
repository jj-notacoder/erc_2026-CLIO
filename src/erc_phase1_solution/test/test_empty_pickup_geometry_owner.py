"""Empty worker wire/identity contracts and original official-model predicates.

The supervisor must stage and pin the official assets before this module runs.
Missing assets fail the fixture; there is no skip or synthetic geometry fallback.
Only the small real-model controls below load meshes; no ROS node is constructed.
"""
from dataclasses import replace
import copy
import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from erc_phase1_solution.empty_pickup_collision import EmptyPickupCollision, ScreenEnvelope
from erc_phase1_solution.empty_pickup_geometry_owner import (
    SCENE_KIND, EmptyPickupGeometryOwner, EmptyPickupGeometryQuery,
    empty_model_signature, validate_empty_pickup_scene,
)
from erc_phase1_solution.geometry_process_protocol import (
    SampleDelta, decode_frame, encode_frame, query_from_wire, query_to_wire,
)
from erc_phase1_solution.motion_profiles import HOME, PREGRASP, RIGHT_HOME
from erc_phase1_solution.pure_geometry_owner import GeometryQuery, PinnedGeometryAssets, RobotGeometryOwner


def scene():
    start = HOME.copy(); start[0] = .35
    return dict(kind=SCENE_KIND, start=start.tolist(), right=RIGHT_HOME.tolist(),
                head=[0., 0.], initial_aperture=.069, open_aperture=.069,
                transition_samples=61)


def query(index=0, **changes):
    current = scene()
    values = dict(request_id=index, epoch=1, source_id='a'*64, model_id='b'*64,
                  scene_id='c'*64, q=current['start'], right=current['right'],
                  head=current['head'], aperture=.069, loaded=False)
    values.update(changes)
    return EmptyPickupGeometryQuery(GeometryQuery.capture(**values)).validated()


def test_wire_roundtrip_retains_operation_exact_bytes_and_signed_zero():
    q = HOME.copy(); q[5] = -0.
    value = query(q=q)
    restored = query_from_wire(decode_frame(encode_frame(query_to_wire(value))))
    assert restored == value and restored.input_sha256 == value.input_sha256
    assert restored.input_sha256 != restored.sample.input_sha256
    assert np.signbit(np.frombuffer(restored.q, dtype=np.float64)[5])
    assert not np.frombuffer(restored.q, dtype=np.float64).flags.writeable


@pytest.mark.parametrize('operation', [None, False, [], {}, 'pickup_payload', 'place'])
def test_unknown_operation_fails_before_geometry(operation):
    with pytest.raises(ValueError, match='operation'):
        replace(query(), operation=operation).validated()


def test_payload_and_nested_queries_are_rejected():
    with pytest.raises(ValueError, match='payload'):
        query(loaded=True)
    value = query()
    with pytest.raises(ValueError, match='ordinary'):
        EmptyPickupGeometryQuery(value).validated()
    wire = query_to_wire(value); wire['sample'] = query_to_wire(value)
    with pytest.raises(ValueError, match='nested'):
        query_from_wire(wire)


@pytest.mark.parametrize('field', ['input_sha256', 'sample', 'extra'])
def test_wire_tampering_cannot_reuse_reply_identity(field):
    wire = query_to_wire(query())
    if field == 'sample':
        wire['sample'] = query_to_wire(query(1).sample)
    elif field == 'input_sha256':
        wire[field] = '0'*64
    else:
        wire[field] = True
    with pytest.raises(ValueError):
        query_from_wire(wire)


@pytest.mark.parametrize('change', [
    {'kind': 'pickup_lift_v1'}, {'bin_scene': {}}, {'table_scene': {}},
    {'attached_corners': []}, {'start': [0.]*7}, {'right': [0.]*8},
    {'head': [float('nan'), 0.]}, {'initial_aperture': -1.00001e-6},
    {'initial_aperture': .06900101}, {'open_aperture': -.000001},
    {'open_aperture': .070}, {'initial_aperture': True},
    {'transition_samples': 2}, {'transition_samples': 1001},
    {'transition_samples': 61.}, {'transition_samples': True},
])
def test_scene_has_only_frozen_empty_geometry_and_original_bounds(change):
    current = scene(); current.update(change)
    with pytest.raises(ValueError):
        validate_empty_pickup_scene(current)


@pytest.mark.parametrize('aperture', [-1e-6, 0., .069, .069+1e-6])
def test_measured_initial_aperture_residual_is_preserved(aperture):
    current = scene(); current['initial_aperture'] = aperture
    admitted = validate_empty_pickup_scene(current)
    assert np.float64(admitted['initial_aperture']).tobytes() == np.float64(aperture).tobytes()
    current['start'][1] = 999.
    assert admitted['start'][1] != 999.


@pytest.fixture(scope='module')
def official_assets():
    # Same sibling official-source layout as the existing URDF tests. The
    # frozen outer runner verifies the expected official revision and digests.
    source = Path(__file__).resolve().parents[1]
    supplied_root = os.environ.get('EMPTY_PICKUP_OFFICIAL_ROOT')
    root = source.parent
    if supplied_root is not None:
        root = Path(supplied_root)
        assert root.is_absolute() and root.is_dir(), 'explicit official root must be absolute and existing'
    urdf = root/'erc_description/urdf/tiago_pro.urdf'
    assert urdf.is_file(), 'staged official URDF is required; no skip'
    packages = {}
    for path in root.rglob('package.xml'):
        name = ET.parse(path).getroot().findtext('name')
        assert name and name not in packages, 'ambiguous staged package inventory'
        packages[name] = path.parent
    packages['realsense2_description'] = Path('/opt/ros/humble/share/realsense2_description')
    description = packages['erc_description']
    required = {urdf, description/'models/collection_bin/meshes/erc_base_collection_bin.STL',
                description/'models/table/meshes/erc_base_table.STL',
                description/'models/table/sdf/erc_table.sdf'}
    for mesh in ET.parse(urdf).getroot().findall('./link/collision/geometry/mesh'):
        uri = mesh.get('filename', '')
        assert uri.startswith('package://')
        package, name = uri[len('package://'):].split('/', 1)
        required.add(packages[package]/name)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    source_files = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob('*')
                    if p.is_file() and p.suffix != '.pyc' and not
                    {'__pycache__', '.pytest_cache', '.ruff_cache'}.intersection(p.parts)}
    assets = PinnedGeometryAssets(source_root=source, source_files=source_files,
        urdf=urdf, bin_mesh=description/'models/collection_bin/meshes/erc_base_collection_bin.STL',
        packages=packages, asset_files={str(p): digest(p) for p in required})
    yield assets
    assets.verify()


@pytest.fixture
def real_owner(official_assets):
    owner = EmptyPickupGeometryOwner(official_assets)
    owner.begin_scene(epoch=1, scene=scene())
    assert type(owner.robot) is RobotGeometryOwner
    assert type(owner.checker) is EmptyPickupCollision
    assert owner.robot._scene is None  # No loaded/PLACE scene was ever installed.
    assert owner.checker._world_geometry_cache is not None
    return owner


def owner_query(owner, index=0, **changes):
    values = dict(source_id=owner.assets.source_id, model_id=owner.assets.model_id,
                  scene_id=owner._scene_id)
    values.update(changes)
    return query(index, **values)


def test_original_checker_and_empty_owner_match_actual_official_samples(real_owner):
    owner = real_owner
    ordinary = EmptyPickupCollision(owner.robot, owner.checker.start,
        owner.checker.right, owner.checker.head, owner.checker.initial_aperture,
        geometry=owner.checker.geometry, screen=owner.checker.screen)
    home = HOME.copy(); home[0] = .35
    pregrasp = PREGRASP.copy(); pregrasp[0] = .35
    samples = [(home, .069), (pregrasp, .069), (home, .035), (home, .069)]
    verdicts = []
    for index, (q, aperture) in enumerate(samples):
        expected = ordinary.sample(q, aperture)
        value = owner_query(owner, index, q=q, aperture=aperture)
        actual = owner.evaluate_independent(value)
        assert actual.matches(value)
        assert actual.verdict is (expected is None)
        assert actual.rejection_update == (None if expected is None else {'reason': expected})
        assert actual.minimum_update is None and actual.table_touched is False
        assert actual.table_intersection is None
        verdicts.append(actual.verdict)
    assert any(verdicts), 'official empty controls must include a clear state'
    assert owner.checker.checked_samples == ordinary.checked_samples
    assert owner.checker.cache_hits == ordinary.cache_hits
    assert owner.checker.cache == ordinary.cache
    assert owner.checker.last_rejection == ordinary.last_rejection
    assert owner.checker.last_rejected_q == ordinary.last_rejected_q
    assert owner.checker.last_rejected_aperture == ordinary.last_rejected_aperture


def test_plain_queries_and_wrong_identity_never_call_sample(real_owner, monkeypatch):
    owner = real_owner
    monkeypatch.setattr(owner.checker, 'sample', lambda *a: pytest.fail('sample reached'))
    with pytest.raises(ValueError, match='requires'):
        owner.evaluate_independent(owner_query(owner).sample)
    for name, value in [('source_id', '1'*64), ('model_id', '2'*64),
                        ('scene_id', '3'*64), ('epoch', 2)]:
        with pytest.raises(RuntimeError, match='identity'):
            owner.evaluate_independent(owner_query(owner, **{name: value}))
    assert owner._queries_seen == 0


def test_frozen_right_and_head_require_exact_bytes(real_owner, monkeypatch):
    owner = real_owner
    monkeypatch.setattr(owner.checker, 'sample', lambda *a: pytest.fail('sample reached'))
    for name, values, zero_index in [('right', owner.checker.right.copy(), 4),
                                     ('head', owner.checker.head.copy(), 0)]:
        assert values[zero_index] == 0.
        values[zero_index] = -0.
        with pytest.raises(RuntimeError, match='frozen'):
            owner.evaluate_independent(owner_query(owner, **{name: values}))
    assert owner._queries_seen == 0


def test_finite_joint_and_aperture_limits_keep_original_uncached_reason(real_owner):
    owner = real_owner
    q = owner.checker.start.copy(); q[0] = owner.robot.chain.lower[0] - .001
    for index, changes in enumerate([{'q': q}, {'aperture': -.001}, {'aperture': .070}]):
        result = owner.evaluate_independent(owner_query(owner, index, **changes))
        assert not result.verdict and result.rejection_update == {'reason': 'empty_pickup_joint_limit'}
    assert owner.checker.checked_samples == 0 and owner.checker.cache_hits == 0
    assert owner.checker.cache == {} and owner.checker.last_rejected_q is None
    assert owner.checker.last_rejected_aperture is None


def test_scene_epoch_cannot_be_replaced_and_request_order_allows_only_forward_gaps(real_owner):
    owner = real_owner
    with pytest.raises(ValueError, match='advance'):
        owner.begin_scene(epoch=1, scene=scene())
    with pytest.raises(RuntimeError, match='immutable'):
        owner.begin_scene(epoch=2, scene=scene())
    for index in (2, 8):
        assert not owner.evaluate_independent(owner_query(owner, index, aperture=-.001)).verdict
    for index in (8, 7, 0):
        with pytest.raises(RuntimeError, match='increasing'):
            owner.evaluate_independent(owner_query(owner, index, aperture=-.001))
    assert owner._queries_seen == 2


@pytest.mark.parametrize('bound', ['maximum_queries', 'maximum_distinct_apertures'])
def test_resource_bound_latches_cancellation_without_running_an_extra_sample(real_owner, bound):
    owner = real_owner; setattr(owner, bound, 1)
    owner.evaluate_independent(owner_query(owner, 0, aperture=-.001))
    with pytest.raises(RuntimeError, match='resource bound'):
        owner.evaluate_independent(owner_query(owner, 1, aperture=-.002))
    assert owner.robot._cancel.is_set() and owner._queries_seen == 1


def test_cancelled_cache_hit_retains_original_counts_and_emits_only_current_reason(real_owner):
    owner = real_owner
    first = owner.evaluate_independent(owner_query(owner))
    assert first.verdict
    owner.checker.last_rejection = 'earlier_candidate'
    success = owner.evaluate_independent(owner_query(owner, 1))
    assert success.verdict and success.rejection_update is None
    assert owner.checker.last_rejection == 'earlier_candidate'
    counts = owner.checker.checked_samples, owner.checker.cache_hits
    owner.cancel()
    cancelled = owner.evaluate_independent(owner_query(owner, 2))
    assert cancelled.rejection_update == {'reason': 'empty_pickup_cancelled'}
    assert not cancelled.verdict
    assert (owner.checker.checked_samples, owner.checker.cache_hits) == counts


def test_geometry_exception_latches_cancel_and_preserves_failure(real_owner, monkeypatch):
    owner = real_owner
    def fail(*args):
        raise ValueError('deliberate original predicate failure')
    monkeypatch.setattr(owner.checker, '_sample_uncached', fail)
    with pytest.raises(ValueError, match='deliberate'):
        owner.evaluate_independent(owner_query(owner))
    assert owner.robot._cancel.is_set()
    result = owner.evaluate_independent(owner_query(owner, 1))
    assert result.rejection_update == {'reason': 'empty_pickup_cancelled'}


def test_real_parent_signature_matches_owner_and_only_transport_hook_is_ignored(real_owner):
    from erc_phase1_solution.manipulation_node import ManipulationNode
    owner = real_owner
    # Actual class/methods with the same actual official models; no ROS ctor.
    parent = ManipulationNode.__new__(ManipulationNode)
    parent.__dict__.update(owner.robot.__dict__)
    checker = EmptyPickupCollision(parent, owner.checker.start, owner.checker.right,
        owner.checker.head, owner.checker.initial_aperture,
        geometry=owner.checker.geometry, screen=owner.checker.screen)
    expected = owner.model_signature()
    assert empty_model_signature(checker, scene()) == expected
    checker._parallel_sequence = lambda *args: pytest.fail('signature invoked transport')
    assert empty_model_signature(checker, scene()) == expected
    checker.context = dict(checker.context, head_positions=np.array([.001, 0.]))
    with pytest.raises(ValueError, match='effective context'):
        empty_model_signature(checker, scene())


def test_real_signature_binds_screen_geometry_and_original_predicate(real_owner, monkeypatch):
    owner = real_owner
    expected = owner.model_signature()
    changed = copy.copy(owner.checker.screen)
    changed.corners = changed.corners.copy(); changed.corners[0, 0] += .001
    monkeypatch.setattr(owner.checker, 'screen', changed)
    assert owner.model_signature() != expected
    monkeypatch.setattr(owner.checker, 'sample', lambda *args: None)
    with pytest.raises(ValueError, match='ordinary empty pickup checker'):
        owner.model_signature()
