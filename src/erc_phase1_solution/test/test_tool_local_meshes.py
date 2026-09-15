"""Prepared tests for immutable tool ownership, eviction and scene order.

No ROS/model assets are needed. Tests are not executed during the live trial.
The scene comparison uses the existing controlled predicate fixture; a matched
real-mesh replay remains a separate prerequisite to integration.
"""
import ast
from candidate_composition_support import restore_motion_source
import copy
import hashlib
import json
from tool_local_ast_support import restore_tool_constructor_keyword
import math
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from erc_phase1_solution.exact_local_bounds import ExactLocalBounds, ModelLocalMesh
from erc_phase1_solution.tool_local_meshes import ToolLocalMeshes, _owner
from erc_phase1_solution import shelf_cradle_geometry as shelf

PACKAGE=Path(__file__).resolve().parents[1]/'erc_phase1_solution'
FIXTURES=Path(__file__).resolve().parent/'fixtures'
ORIGINAL_SCENE=FIXTURES/'tool_local_original_scene.py'
ORIGINAL_SHELF=FIXTURES/'tool_local_original_shelf.py'
SCOPE=FIXTURES/'tool_local_original_scope.json'
SCOPE_SHA='8b0e2b425cc40382b9fe0e03c69d34becae313ad24350b4ad3ab90b84f13e516'

def triangles():
    return np.array([[[-.2,-.2,.9],[.2,-.2,1.1],[0.,.2,1.]],
                     [[-.1,-.1,1.],[.1,.1,.9],[0.,0.,1.2]]],dtype=np.float64)

def signature(a):
    a=np.asarray(a)
    return a.shape,a.dtype.str,a.tobytes()

def geometry(monkeypatch,*,immutable=True,maximum_entries=64,maximum_bytes=32*1024*1024):
    g=object.__new__(shelf.ShelfCradleGeometry)
    g.model=object();g.mimics={};g.watertight={'tool':True};g._local_cache={}
    g._local_mesh_cache=ToolLocalMeshes(maximum_entries=maximum_entries,maximum_bytes=maximum_bytes) if immutable else None
    calls=[]
    def forward(values):
        calls.append(tuple(values)); t=np.eye(4)
        if values:t[0,3]=values[0]
        return t
    g.chains={'tool':NS(active_names=['gripper_left_finger_joint'],forward=forward)}
    g.grasp_chain=NS(active_names=[],forward=forward)
    monkeypatch.setattr(shelf,'world_gripper_surfaces',lambda model,transforms:
        {'tool':triangles() @ transforms['tool'][:3,:3].T+transforms['tool'][:3,3]})
    return g,calls

def test_existing_immutable_payload_reuses_exact_owner_and_rebuilds_bounds():
    original=ModelLocalMesh(triangles()); surface,low,high=original.snapshot()
    rebuilt=ModelLocalMesh.from_immutable_surface(surface)
    assert rebuilt.matches(surface) and _owner(rebuilt.snapshot()[0]) is _owner(surface)
    assert [signature(a) for a in rebuilt.snapshot()]==[signature(a) for a in (surface,low,high)]
    surface.shape=(18,)
    assert rebuilt.snapshot()[0].shape==(2,3,3)
    for value in rebuilt.snapshot():
        with pytest.raises(ValueError):value.setflags(write=True)

@pytest.mark.parametrize('kind',['mutable','readonly_owned','readonly_alias','partial','subclass',
    'dtype','foreign_endian','empty','wrong_shape','nan','infinity'])
def test_reconstruction_rejects_unowned_partial_or_invalid_surface(kind):
    raw=triangles()
    surface=ModelLocalMesh(raw).snapshot()[0]
    if kind=='mutable':surface=raw
    elif kind=='readonly_owned':surface=raw;surface.setflags(write=False)
    elif kind=='readonly_alias':surface=raw.view();surface.setflags(write=False)
    elif kind=='partial':surface=surface[:1]
    elif kind=='subclass':
        class Custom(np.ndarray):pass
        surface=surface.view(Custom)
    elif kind=='dtype':surface=surface.view(np.uint64)
    elif kind=='foreign_endian':surface=np.frombuffer(raw.astype('>f8').tobytes(),dtype='>f8').reshape(raw.shape)
    elif kind=='empty':surface=np.frombuffer(b'',dtype=np.float64).reshape((0,3,3))
    elif kind=='wrong_shape':surface=surface.reshape(3,6)
    elif kind in ('nan','infinity'):
        raw[0,0,0]=math.nan if kind=='nan' else math.inf
        surface=np.frombuffer(raw.tobytes(),dtype=np.float64).reshape(raw.shape)
    with pytest.raises(ValueError):ModelLocalMesh.from_immutable_surface(surface)

def test_default_raw_cache_keeps_its_existing_mutation_behavior(monkeypatch):
    g,calls=geometry(monkeypatch,immutable=False)
    first=g.local_surfaces(.017); first['tool'][0,0,0]=19.
    second=g.local_surfaces(.017)
    assert second is first and second['tool'][0,0,0]==19.
    assert len(calls)==2 and g.model_local_surface(second['tool']) is None

def test_opt_in_values_match_raw_and_metadata_cannot_poison_later_calls(monkeypatch):
    raw,_=geometry(monkeypatch,immutable=False);frozen,calls=geometry(monkeypatch)
    expected=raw.local_surfaces(.017)
    first=frozen.local_surfaces(.017)
    assert signature(first['tool'])==signature(expected['tool'])
    surface=first['tool'];model=frozen.model_local_surface(surface)
    assert model is not None and model.matches(surface)
    with pytest.raises(ValueError):surface[0,0,0]=19.
    surface.shape=(18,);surface.base.shape=(18,)
    first['extra']=triangles();first.pop('tool')
    second=frozen.local_surfaces(.017)
    assert list(second)==['tool'] and signature(second['tool'])==signature(expected['tool'])
    assert len(calls)==2 and frozen.model_local_surface(second['tool']).matches(second['tool'])

def test_exact_aperture_and_explicit_finger_key_semantics_remain(monkeypatch):
    g,calls=geometry(monkeypatch)
    a=.017;b=math.nextafter(a,math.inf)
    first=g.local_surfaces(a);g.local_surfaces(b)
    assert len(g._local_cache)==2 and len(calls)==4
    same=g.local_surfaces(.5,{'gripper_left_finger_joint':a})
    assert len(calls)==4 and signature(same['tool'])==signature(first['tool'])
    with pytest.raises(ValueError,match='nonfinite'):
        g.local_surfaces(math.nan)

def test_handle_eviction_preserves_geometry_cache_without_recomputing_fk(monkeypatch):
    g,calls=geometry(monkeypatch,maximum_entries=1)
    first=g.local_surfaces(.017);old_owner=_owner(first['tool'])
    expected=signature(first['tool']);g.local_surfaces(.069)
    assert len(g._local_mesh_cache._entries)==1
    assert g.model_local_surface(first['tool']) is None
    before=len(calls);again=g.local_surfaces(.017)
    assert len(calls)==before and signature(again['tool'])==expected
    assert _owner(again['tool']) is old_owner
    assert g.model_local_surface(again['tool']).matches(again['tool'])
    assert len(g._local_mesh_cache._entries)==1
    assert g._local_mesh_cache._bytes<=g._local_mesh_cache.maximum_bytes

def test_byte_capacity_evicts_only_added_registry(monkeypatch):
    weight=triangles().nbytes+48
    g,calls=geometry(monkeypatch,maximum_bytes=weight)
    a=g.local_surfaces(.017);b=g.local_surfaces(.069)
    assert len(g._local_cache)==2 and len(g._local_mesh_cache._entries)==1
    assert g._local_mesh_cache._bytes==weight
    assert not g.model_local_surface(a['tool']) and g.model_local_surface(b['tool'])

def test_too_large_or_nonfinite_producer_output_keeps_raw_fallback():
    reg=ToolLocalMeshes(maximum_bytes=1)
    raw=triangles();packed=reg.freeze({'tool':raw})
    assert packed['tool'] is raw and reg.materialize(packed)['tool'] is raw
    assert not reg._entries and reg._bytes==0
    reg=ToolLocalMeshes();bad=triangles();bad[0,0,0]=math.nan
    packed=reg.freeze({'tool':bad})
    assert reg.materialize(packed)['tool'] is bad and reg.model_for_surface(bad) is None

def test_frozen_payload_does_not_alias_producer_or_prior_metadata():
    registry=ToolLocalMeshes()
    raw=triangles();raw[0,0,0]=-0.
    expected=signature(raw)
    cached=registry.freeze({'tool':raw})
    raw[:]=81.
    first=registry.materialize(cached)['tool']
    assert signature(first)==expected
    model=registry.model_for_surface(first)
    first.dtype=np.uint64
    assert model.matches(registry.materialize(cached)['tool'])
    assert signature(registry.materialize(cached)['tool'])==expected

@pytest.mark.parametrize('value',[0,1,None,'yes'])
def test_bad_opt_in_flag_is_rejected_before_loading_model(monkeypatch,value):
    def forbidden(*args):raise AssertionError('model loading must not occur')
    monkeypatch.setattr(shelf,'load_left_gripper_collision_model',forbidden)
    with pytest.raises(ValueError,match='Boolean'):
        shelf.ShelfCradleGeometry(Path('unused.urdf'),forbidden,immutable_local=value)

def test_custom_producer_and_foreign_model_owner_cannot_get_handle(monkeypatch):
    a,_=geometry(monkeypatch);b,_=geometry(monkeypatch)
    surface_a=a.local_surfaces(.017)['tool'];surface_b=b.local_surfaces(.017)['tool']
    assert signature(surface_a)==signature(surface_b)
    assert a.model_local_surface(surface_b) is None
    a.local_surfaces=lambda aperture:{'tool':surface_a}
    assert a.model_local_surface(surface_a) is None

@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_registry_rejects_bad_capacity(value):
    with pytest.raises(ValueError):ToolLocalMeshes(maximum_entries=value)
    with pytest.raises(ValueError):ToolLocalMeshes(maximum_bytes=value)

def test_handle_path_does_not_enter_raw_content_cache(monkeypatch):
    g,_=geometry(monkeypatch)
    surface=g.local_surfaces(.017)['tool'];handle=g.model_local_surface(surface)
    class NoLookup(dict):
        def get(self,*a):raise AssertionError('raw content lookup')
    cache=ExactLocalBounds();cache._entries=NoLookup()
    assert signature(cache.capture(handle)[0])==signature(surface)
    assert cache._bytes==0 and not cache._entries

def tool_for_scene(fixture):
    g=object.__new__(shelf.ShelfCradleGeometry)
    g._local_mesh_cache=ToolLocalMeshes()
    g._local_cache={(('gripper_left_finger_joint',.017),):g._local_mesh_cache.freeze(fixture.tools)}
    g.watertight={name:True for name in fixture.tools}
    return g

@pytest.mark.parametrize('failure',[None,'tool_bin','tool_table'])
@pytest.mark.parametrize('custom',[False,True])
def test_actual_scene_preserves_query_order_world_values_and_first_rejection(failure,custom):
    import test_scene_world_bounds_differential as shared
    from erc_phase1_solution.scene_checked_place import NominalBinObstacle,TableSceneObstacle
    records=[]
    for path in (ORIGINAL_SCENE,PACKAGE/'scene_checked_place.py'):
        f=shared.Fixture(compound=True,screen=False)
        f.tool=tool_for_scene(f)
        seen=[];counts={'bin':0,'table':0}
        if not custom:
            f.obstacle=object.__new__(NominalBinObstacle);f.table=object.__new__(TableSceneObstacle)
            f.obstacle.book_intersects=lambda value:f.call('book_bin',signature(value))
            f.table.intersects_box=lambda value:f.call('book_table',signature(value))
            f.table.last_intersection='fixture_solid'
        def query(label,surface,transform,watertight):
            counts[label]+=1
            seen.append((label,type(surface)))
            f.call(label,signature(surface),signature(transform),watertight)
            return counts[label]==3 and failure=='tool_'+label
        f.obstacle.intersects=lambda *a:query('bin',*a)
        f.table.intersects=lambda *a:query('table',*a)
        cls=shared.classes(path,f)['PlaceSceneChecker']
        scene=cls(f.node,f.obstacle,f.tool,f.attached,f.table,None)
        result=scene.sample(np.zeros(8),.017,True)
        records.append((result,scene.last_rejection,list(f.trace),seen))
    assert records[0][:3]==records[1][:3]
    if custom:
        assert all(kind is np.ndarray for _,kind in records[1][3])
    else:
        assert any(kind is ModelLocalMesh for _,kind in records[1][3])
        assert all(kind is np.ndarray for _,kind in records[0][3])

def test_all_normal_shared_tool_creation_sites_opt_in_and_default_remains_raw():
    found=[]
    for filename in ('empty_pickup_collision.py','open_gripper_approach.py',
                     'manipulation_node.py','scene_checked_place.py','shelf_cradle_geometry.py'):
        tree=ast.parse((PACKAGE/filename).read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id=='ShelfCradleGeometry']
        assert len(calls)==1,filename
        assert any(k.arg=='immutable_local' and isinstance(k.value,ast.Constant) and k.value.value is True for k in calls[0].keywords)
        found.append(filename)
    cls=next(n for n in ast.parse((PACKAGE/'shelf_cradle_geometry.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ShelfCradleGeometry')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assert [(a.arg,d.value) for a,d in zip(init.args.kwonlyargs,init.args.kw_defaults)]==[('immutable_local',False)]

def test_no_node_pick_or_open_geometry_changes_beyond_constructor_keyword():
    assert hashlib.sha256(SCOPE.read_bytes()).hexdigest()==SCOPE_SHA
    originals=json.loads(SCOPE.read_text())['whole_modules']
    for filename in ('empty_pickup_collision.py','open_gripper_approach.py','manipulation_node.py'):
        source=(PACKAGE/filename).read_text()
        if filename=='empty_pickup_collision.py':
            from empty_pickup_snapshot_support import restore_empty_pickup_snapshot
            source=restore_empty_pickup_snapshot(source)
        changed=ast.parse(restore_motion_source(source) if filename=='manipulation_node.py' else source)
        restored=restore_tool_constructor_keyword(changed)
        actual=hashlib.sha256(ast.dump(restored,include_attributes=False).encode()).hexdigest()
        assert actual==originals[filename]['ast_sha256'],filename


def test_exact_original_scene_and_shelf_fixtures_are_pinned():
    assert hashlib.sha256(SCOPE.read_bytes()).hexdigest()==SCOPE_SHA
    for filename,expected in json.loads(SCOPE.read_text())['fixtures'].items():
        assert hashlib.sha256((FIXTURES/filename).read_bytes()).hexdigest()==expected


def test_original_tool_arithmetic_and_aperture_key_are_exact_ast():
    def nodes(path):
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ShelfCradleGeometry')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='local_surfaces')
        names={'values','key','transforms','grasp','surfaces'}
        return [ast.dump(n,include_attributes=False) for n in fn.body if
            (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))
            or (isinstance(n,ast.FunctionDef) and n.name=='value')]
    assert nodes(ORIGINAL_SHELF)==nodes(PACKAGE/'shelf_cradle_geometry.py')
