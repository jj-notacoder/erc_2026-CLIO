"""Tiny immutable models exercise actual registered-shelf predicates and grids."""
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution import empty_shelf_bounds as current
from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
from erc_phase1_solution.kinematics import CollisionMesh, Joint, URDFChain
from erc_phase1_solution.lift_first_extraction import RelativeShelfBay
from erc_phase1_solution.pure_geometry_owner import RobotGeometryOwner
from erc_phase1_solution.rigid_palm_preflight import LEFT_GRIPPER_COLLISION_LINKS
from erc_phase1_solution.shelf_cradle_geometry import ShelfCradleGeometry
from erc_phase1_solution.tool_local_meshes import ToolLocalMeshes

PARENT = Path(__file__).parent/'fixtures/registered_shelf_original.py'
spec = importlib.util.spec_from_file_location('registered_shelf_original', PARENT)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def triangles(x=.30, z=.60):
    return np.array([[[x,-.01,z], [x+.01,.01,z], [x,.01,z+.01]]], dtype=float)


def mesh(link, surface):
    model = ModelLocalMesh(surface)
    return CollisionMesh(link, model.snapshot()[0],
        np.array([surface.min(axis=(0,1)),surface.max(axis=(0,1))]), False, model)


def chain(side, count):
    names = ['torso_lift_joint']
    joints = [Joint(names[0], 'base_link', 'torso_lift_link', 'prismatic',
        np.eye(4), np.array([0.,0.,1.]), 0., .35)]
    for index in range(1, count+1):
        name = f'{side}_{index}_joint'; names.append(name)
        joints.append(Joint(name, joints[-1].child, f'{side}_{index}_link',
            'prismatic', np.eye(4), np.array([1.,0.,0.]), -3., 3.))
    if side == 'head':
        joints.append(Joint('camera_fixed', joints[-1].child, 'head_front_camera_link',
            'fixed', np.eye(4), np.array([0.,0.,1.]), 0., 0.))
    return URDFChain(joints, names)


def fixture(*, maximum_entries=4096):
    node = object.__new__(RobotGeometryOwner)
    node.chain, node.right_chain, node.head_chain = chain('arm_left',7), chain('arm_right',7), chain('head',2)
    node._cancel = threading.Event()
    node.carried_transition_samples = 61
    node.carried_book_dimensions = np.array([.16,.02,.25])
    node.carried_collision_meshes = tuple(mesh(name,triangles()) for name in
        ('arm_left_1_link','arm_right_1_link','head_1_link'))
    geometry = object.__new__(ShelfCradleGeometry)
    geometry._local_mesh_cache = ToolLocalMeshes()
    aperture_key = (('gripper_left_finger_joint',.069),)
    geometry._local_cache = {aperture_key:geometry._local_mesh_cache.freeze(
        {name:triangles() for name in LEFT_GRIPPER_COLLISION_LINKS})}
    guard = NS(geometry=geometry, open_aperture=.069,
        context={'right_positions':np.zeros(7),'head_positions':np.zeros(2)})
    bay = RelativeShelfBay(marker_center_base=np.array([.605,-.006,2.26]),
        inward_axis_base=np.array([1.,0.,0.]),physical_column=3,
        lateral_uncertainty_m=.200,roof_uncertainty_m=.010,
        assume_upright_supported=True,source='registered tiny immutable fixture')
    front = np.array([.670,-.056,.604])
    cache = current.ExactShelfSampleCache(node, guard, maximum_entries=maximum_entries)
    def bounds(*, cached=True, **kwargs):
        cls = current.EmptyShelfBounds if cached else baseline.EmptyShelfBounds
        options = dict(sample_cache=cache) if cached else {}
        return cls(node,front,None,guard,bay,**options,**kwargs)
    return NS(node=node,guard=guard,bay=bay,front=front,cache=cache,bounds=bounds)


def pose(x=0.):
    q = np.zeros(8); q[1] = x
    return q


def metrics(bounds):
    return (bounds.samples, *(np.float64(getattr(bounds,'minimum_'+name)).tobytes()
        for name in ('floor','roof','side','back')))


def assert_edge_parity(f, start, end, *, allow_entry):
    old, new = f.bounds(cached=False), f.bounds()
    expected = old.edge(start,end,allow_entry=allow_entry)
    assert new.edge(start,end,allow_entry=allow_entry) == expected
    assert metrics(new) == metrics(old)
    before = f.cache.hits
    admitted = f.bounds()
    assert admitted.edge(start,end,allow_entry=allow_entry) == expected
    assert metrics(admitted) == metrics(old)
    assert f.cache.hits-before == old.samples
    return old, new, admitted


@pytest.mark.parametrize('start,end,entry',[(0.,.2,False),(.30,.34,True),(.20,.35,False),(.30,.65,True),(0.,0.,False)])
def test_repeated_exact_edges_preserve_verdict_full_grid_and_metric_bytes(start,end,entry):
    f=fixture();old,_,_=assert_edge_parity(f,pose(start),pose(end),allow_entry=entry)
    if old.edge(pose(start),pose(end),allow_entry=entry) is None:
        assert old.samples==122  # unchanged61 samples, including zero-length edges


def test_pinned_parent_bytes():
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == '06e6b8d36a1ec607c131de1fe23967ecba5f8a168ad4a22c589b0c776d31f6ed'


def test_final_metrics_exclude_rejected_search_alternatives():
    f=fixture();search=f.bounds()
    assert search.edge(pose(.30),pose(.65),allow_entry=True).startswith('empty_bay_clearance:')
    assert search.minimum_back<0.
    assert search.edge(pose(.30),pose(.34),allow_entry=True) is None
    fresh=f.bounds();old=f.bounds(cached=False)
    assert fresh.edge(pose(.30),pose(.34),allow_entry=True) is None
    old.edge(pose(.30),pose(.34),allow_entry=True)
    assert metrics(fresh)==metrics(old) and fresh.minimum_back>0.


def test_allow_entry_is_part_of_exact_query():
    f=fixture();bounds=f.bounds();q=pose(.32)
    assert bounds.sample(q,True) is None
    hits=f.cache.hits
    assert bounds.sample(q,False).startswith('empty_setup_enters_shelf:')
    assert f.cache.hits==hits


def test_single_bit_pose_change_is_a_miss():
    f=fixture();bounds=f.bounds();q=pose(.32)
    bounds.sample(q,True);before=f.cache.hits
    changed=q.copy();changed[1]=np.nextafter(changed[1],np.inf)
    bounds.sample(changed,True)
    assert f.cache.hits==before


@pytest.mark.parametrize('change',['right','head','margin','plane','floor','roof','side','uncertainty','tool','robot','chain'])
def test_context_or_model_change_cannot_hit_old_result(change):
    f=fixture();bounds=f.bounds();q=pose(.32)
    bounds.sample(q,True);before=f.cache.hits
    if change in ('right','head'):
        f.guard.context['right_positions' if change=='right' else 'head_positions'][0]+=.01
    elif change=='margin':bounds.margin+=.001
    elif change=='plane':bounds.plane_point[0]+=.001
    elif change in ('floor','roof'):setattr(bounds,change,getattr(bounds,change)+.001)
    elif change=='side':bounds.side_max-=.001
    elif change=='uncertainty':bounds.normal_uncertainty_m+=.001
    elif change=='tool':
        locals=dict(bounds.local);name=next(iter(locals))
        frozen=f.guard.geometry._local_mesh_cache.freeze({name:triangles(z=.48)})
        locals[name]=f.guard.geometry._local_mesh_cache.materialize(frozen)[name]
        bounds.local=locals
    elif change=='robot':
        values=list(f.node.carried_collision_meshes)
        values[0]=mesh(values[0].link,triangles(z=.48));f.node.carried_collision_meshes=tuple(values)
    else:f.node.chain.joints[1].origin[0,3]+=.02  # in-place model mutation changes current FK
    bounds.sample(q,True)
    assert f.cache.hits==before


def test_custom_or_mutable_producer_falls_through_and_drops_old_entries():
    f=fixture();bounds=f.bounds();q=pose(.32)
    bounds.sample(q,True);assert f.cache.entries
    original=f.node._world_collision_surfaces
    calls=[]
    f.node._world_collision_surfaces=lambda *a,**kw:(calls.append(True) or original(*a,**kw))
    before=f.cache.hits
    assert bounds.sample(q,True) is None
    assert calls==[True] and f.cache.hits==before and not f.cache.entries
    del f.node._world_collision_surfaces
    values=list(f.node.carried_collision_meshes)
    values[0]=replace(values[0],triangles=values[0].triangles.copy())
    f.node.carried_collision_meshes=tuple(values)
    assert bounds.sample(q,True) is None and not f.cache.entries


@pytest.mark.parametrize('change',['raw_tool','tool_resolver','tool_order','robot_order','registered_lip','left_axis','inward_axis','aperture'])
def test_additional_exact_surface_identity_changes_invalidate(change):
    f=fixture();bounds=f.bounds();q=pose(.32)
    bounds.sample(q,True);before=f.cache.hits
    if change=='raw_tool':
        name=next(iter(bounds.local));bounds.local[name]=bounds.local[name].copy()
        bounds.local[name][...,2]-=.2
    elif change=='tool_resolver':
        original=f.guard.geometry.model_local_surface
        f.guard.geometry.model_local_surface=lambda surface:original(surface)
    elif change=='tool_order':bounds.local=dict(reversed(tuple(bounds.local.items())))
    elif change=='robot_order':f.node.carried_collision_meshes=tuple(reversed(f.node.carried_collision_meshes))
    elif change=='registered_lip':bounds.registered_lip[0]+=.001
    elif change=='left_axis':bounds.left[0]+=.001
    elif change=='inward_axis':bounds.inward[1]+=.001
    else:f.guard.open_aperture=np.nextafter(f.guard.open_aperture,0.)
    bounds.sample(q,True)
    assert f.cache.hits==before


def test_context_change_during_miss_cannot_publish_old_result_under_new_context():
    f=fixture();bounds=f.bounds();q=pose(.32)
    # The production evaluator is still called exactly; a test observer changes
    # only the final identity read to emulate mutation during that computation.
    identity=f.cache._identity;calls=[]
    def changed(*args):
        calls.append(True)
        if len(calls)==2:bounds.floor+=.2
        return identity(*args)
    f.cache._identity=changed
    assert bounds.sample(q,True) is None
    assert not f.cache.entries
    assert bounds.sample(q,True).startswith('empty_bay_clearance:')
    assert f.cache.hits==0


@pytest.mark.parametrize('name',['marker','left','inward','plane_point','registered_lip'])
def test_same_vector_bytes_with_different_shape_do_not_hit(name):
    f=fixture();bounds=f.bounds();q=pose(.32)
    bounds.sample(q,True);before=f.cache.hits
    setattr(bounds,name,getattr(bounds,name).reshape(1,3))
    assert f.cache.key(bounds,q,True) is None
    assert not f.cache.entries and f.cache.hits==before


@pytest.mark.parametrize('name',['margin','side_min'])
def test_equal_numeric_string_does_not_hide_original_scalar_type_error(name):
    f=fixture();bounds=f.bounds();q=pose(.32)
    bounds.sample(q,True);before=f.cache.hits
    old=f.bounds(cached=False)
    for value in (old,bounds):setattr(value,name,str(getattr(value,name)))
    with pytest.raises(TypeError):old.sample(q,True)
    with pytest.raises(TypeError):bounds.sample(q,True)
    assert f.cache.hits==before and not f.cache.entries


@pytest.mark.parametrize('value',[True,False,'0.015',np.array(.015),np.nan,np.inf,-np.inf])
def test_unsupported_scalar_metadata_is_never_a_cache_identity(value):
    f=fixture();bounds=f.bounds();q=pose(.32)
    bounds.sample(q,True);before=f.cache.hits
    bounds.margin=value
    assert f.cache.key(bounds,q,True) is None
    assert f.cache.hits==before and not f.cache.entries


def test_coarse_success_never_substitutes_for_dense_sample_grid():
    f=fixture();bounds=f.bounds();start,end=pose(0.),pose(.3)
    # The final61 grid must still reach the registered plane between the
    # seven coarse eighth probes and the endpoint.
    for fraction in (.125,.25,.375,.5,.625,.75,.875):
        assert bounds.sample(start+(end-start)*fraction,False) is None
    old=f.bounds(cached=False);final=f.bounds()
    expected=old.edge(start,end,allow_entry=False)
    assert expected.startswith('empty_setup_enters_shelf:')
    assert final.edge(start,end,allow_entry=False)==expected
    assert metrics(final)==metrics(old)


def test_cancelled_cached_sample_stops_before_hit_or_metrics_replay():
    f=fixture();bounds=f.bounds();q=pose(.32)
    bounds.sample(q,True);before=(f.cache.hits,metrics(bounds))
    f.node._cancel.set()
    with pytest.raises(RuntimeError,match='cancelled'):bounds.sample(q,True)
    assert (f.cache.hits,metrics(bounds))==before


def test_cancellation_is_checked_during_replayed_dense_edge():
    f=fixture();f.bounds().edge(pose(.30),pose(.34),allow_entry=True)
    class CancelOnCheck:
        calls=0
        def is_set(self):
            self.calls+=1
            return self.calls>=8
    f.node._cancel=CancelOnCheck();final=f.bounds()
    with pytest.raises(RuntimeError,match='cancelled'):
        final.edge(pose(.30),pose(.34),allow_entry=True)
    assert 0<final.samples<61


def test_bounded_eviction_only_recomputes_original_predicate():
    f=fixture(maximum_entries=2);bounds=f.bounds()
    for x in (.30,.31,.32):assert bounds.sample(pose(x),True) is None
    assert len(f.cache.entries)==2
    before=f.cache.hits
    assert bounds.sample(pose(.30),True) is None
    assert f.cache.hits==before and len(f.cache.entries)==2


def test_unshared_candidate_cannot_reuse_prior_sample():
    first,second=fixture(),fixture()
    first.bounds().sample(pose(.32),True)
    second.bounds().sample(pose(.32),True)
    assert first.cache.hits==second.cache.hits==0


def test_foreign_guard_or_planning_thread_cannot_reuse_cache():
    f=fixture();f.bounds().sample(pose(.32),True);before=f.cache.hits
    guard=NS(**vars(f.guard))
    other=current.EmptyShelfBounds(f.node,f.front,None,guard,f.bay,sample_cache=f.cache)
    assert other.sample(pose(.32),True) is None and f.cache.hits==before
    f.bounds().sample(pose(.32),True);before=f.cache.hits
    errors=[]
    def run():
        try:assert f.bounds().sample(pose(.32),True) is None
        except BaseException as error:errors.append(error)
    thread=threading.Thread(target=run);thread.start();thread.join()
    assert not errors and f.cache.hits==before
