"""Prepared pure cache/actual-obstacle differentials, no ROS or model assets."""
import ast
import copy
import importlib.util
import itertools
import math
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
BASELINE=HERE/'fixtures'
PACKAGE=HERE.parent/'erc_phase1_solution'
spec=importlib.util.spec_from_file_location('exact_bounds_candidate',PACKAGE/'exact_local_bounds.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
ExactLocalBounds=module.ExactLocalBounds


def surface():
    return np.array([[[-.2,-.2,-.2],[.2,-.2,.2],[0.,.2,0.]],
                     [[-.1,-.1,.1],[.1,.1,-.1],[0.,0.,.2]]],dtype=np.float64)


def signature(a):
    x=np.asarray(a)
    return (x.dtype.str,x.shape,x.tobytes())


def test_exact_content_hit_uses_immutable_snapshot_and_bounds():
    source=surface();c=ExactLocalBounds();a=c.capture(source);b=c.capture(source.copy())
    assert a is b
    assert signature(a[0])==signature(source)
    assert signature(a[1])==signature(source.min(axis=(0,1)))
    assert signature(a[2])==signature(source.max(axis=(0,1)))
    for array in a:
        assert not array.flags.writeable
        with pytest.raises(ValueError): array.setflags(write=True)
    source+=10.
    assert np.max(a[0])<1.
    new=c.capture(source)
    assert new is not a and signature(new[0])==signature(source)


def test_owner_modified_through_external_alias_is_a_new_content_key():
    source=surface();alias=source.view();c=ExactLocalBounds()
    first=c.capture(source);alias[0,0,0]=12.
    second=c.capture(source)
    assert first is not second and second[2][0]==12.
    assert c.capture(alias) is None


def test_signed_zero_and_shape_are_part_of_exact_content():
    c=ExactLocalBounds();a=np.zeros((2,3,3));a[0,0,0]=-0.
    one=c.capture(a);b=a.copy();b[0,0,0]=0.;two=c.capture(b)
    assert one is not two
    for original,entry in ((a,one),(b,two)):
        assert signature(entry[1])==signature(original.min(axis=(0,1)))
        assert signature(entry[2])==signature(original.max(axis=(0,1)))
    assert c.capture(np.zeros((3,3))) is None


class Subclass(np.ndarray):
    pass


@pytest.mark.parametrize('kind',['view','noncontiguous','subclass','float32','foreign_endian','empty','badshape','nan','inf','list'])
def test_unsupported_or_nonfinite_does_not_enter_cache(kind):
    a=surface()
    if kind=='view': a=a.view()
    elif kind=='noncontiguous': a=a[:,::-1,:]
    elif kind=='subclass': a=a.view(Subclass)
    elif kind=='float32': a=a.astype(np.float32)
    elif kind=='foreign_endian': a=a.astype('>f8')
    elif kind=='empty': a=np.empty((0,3,3))
    elif kind=='badshape': a=np.zeros((3,3))
    elif kind=='nan': a[0,0,0]=np.nan
    elif kind=='inf': a[0,0,0]=np.inf
    elif kind=='list': a=a.tolist()
    c=ExactLocalBounds(); assert c.capture(a) is None
    assert not c._entries and c._bytes==0


def test_lru_entry_and_byte_caps_and_surviving_evicted_snapshot():
    a=surface();weight=a.nbytes+48
    c=ExactLocalBounds(maximum_entries=2,maximum_bytes=2*weight)
    first=c.capture(a);second=c.capture(a+1.)
    assert c.capture(a.copy()) is first  # first is now most recent
    c.capture(a+2.)
    assert len(c._entries)==2 and c._bytes==2*weight
    assert c.capture(a+1.) is not second
    assert signature(first[0])==signature(a)
    tiny=ExactLocalBounds(maximum_bytes=weight-1)
    assert tiny.capture(a) is None and tiny._bytes==0


@pytest.mark.parametrize('kw',[{'maximum_entries':0},{'maximum_entries':True},{'maximum_bytes':0},{'maximum_bytes':1.5}])
def test_invalid_capacity_rejected(kw):
    with pytest.raises(ValueError): ExactLocalBounds(**kw)


def obstacle_class(path,kind,predicate):
    source=ast.parse(path.read_text(encoding='utf-8'))
    name='TableSceneObstacle' if kind=='table' else 'NominalBinObstacle'
    names={name,'_finite','_rigid'}
    body=[copy.deepcopy(n) for n in source.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    env=dict(np=np,math=math,itertools=itertools,ExactLocalBounds=ExactLocalBounds,
             oriented_box_intersects_triangles=predicate)
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*body],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),env)
    return env[name]


def fixture(kind,*,cache,predicate,monkeypatch):
    filename='table_scene.py' if kind=='table' else 'scene_checked_place.py'
    path=PACKAGE/filename if cache else BASELINE/('local_bounds_original_'+kind+'.py')
    if kind=='table':
        fake=ModuleType('erc_phase1_solution.kinematics');fake.oriented_box_intersects_triangles=predicate
        monkeypatch.setitem(sys.modules,'erc_phase1_solution.kinematics',fake)
    cls=obstacle_class(path,kind,predicate);o=cls.__new__(cls)
    o.rotation=np.eye(3);o.origin=np.zeros(3);o.bounds=np.array([[-.5]*3,[.5]*3])
    corners=np.array(list(itertools.product((-.5,.5),repeat=3)))
    o.material_corners=[corners];o.material_bounds=[o.bounds]
    o.solids=[('fixture',o.bounds,corners)];o.last_intersection=None
    if cache:o._local_bounds=ExactLocalBounds()
    return o


def run(kind,source,transform,monkeypatch,*,cache,predicate_value=True):
    calls=[]
    def predicate(corners,world,*,closed_surface):
        calls.append((signature(corners),signature(world),closed_surface));return predicate_value
    o=fixture(kind,cache=cache,predicate=predicate,monkeypatch=monkeypatch)
    try: result=('return',o.intersects(source,transform,True))
    except Exception as exc:result=('raise',type(exc).__name__,str(exc))
    return result,calls,o


@pytest.mark.parametrize('kind',['table','bin'])
@pytest.mark.parametrize('shift',[0.,.4,5.])
@pytest.mark.parametrize('predicate_value',[False,True])
def test_original_and_candidate_actual_intersects_match(kind,shift,predicate_value,monkeypatch):
    t=np.eye(4);angle=.6;c,s=math.cos(angle),math.sin(angle)
    t[:3,:3]=[[c,-s,0.],[s,c,0.],[0.,0.,1.]];t[0,3]=shift
    baseline=run(kind,surface(),t,monkeypatch,cache=False,predicate_value=predicate_value)
    changed=run(kind,surface(),t,monkeypatch,cache=True,predicate_value=predicate_value)
    assert baseline[:2]==changed[:2]
    assert baseline[2].last_intersection==changed[2].last_intersection


@pytest.mark.parametrize('kind',['table','bin'])
def test_cached_bounds_and_narrow_world_use_identical_snapshot(kind,monkeypatch):
    caller=surface();calls=[]
    def predicate(corners,world,**kwargs): calls.append(world.copy());return True
    o=fixture(kind,cache=True,predicate=predicate,monkeypatch=monkeypatch)
    original=o._local_bounds.capture
    def capture_then_mutate(value):
        result=original(value);value+=100.;return result
    o._local_bounds.capture=capture_then_mutate
    before=caller.copy()
    assert o.intersects(caller,np.eye(4),True)
    np.testing.assert_array_equal(calls[0],before)
    # A later call sees the changed content, and clears the fixture obstacle.
    o._local_bounds.capture=original
    assert not o.intersects(caller,np.eye(4),True)


@pytest.mark.parametrize('kind',['table','bin'])
@pytest.mark.parametrize('variant',['view','float32','subclass','noncontiguous','empty','nan'])
def test_original_fallback_and_exception_paths_preserved(kind,variant,monkeypatch):
    a=surface()
    if variant=='view':a=a.view()
    elif variant=='float32':a=a.astype(np.float32)
    elif variant=='subclass':a=a.view(Subclass)
    elif variant=='noncontiguous':a=a[:,::-1,:]
    elif variant=='empty':a=np.empty((0,3,3))
    else:a[0,0,0]=np.nan
    old=run(kind,a,np.eye(4),monkeypatch,cache=False)
    new=run(kind,a,np.eye(4),monkeypatch,cache=True)
    assert old[:2]==new[:2]
    assert not new[2]._local_bounds._entries


def test_table_surface_then_transform_validation_order_preserved(monkeypatch):
    class BadTransform:
        def __array__(self,*args,**kwargs): raise RuntimeError('transform visited')
    for source,expected in [(np.zeros((3,3)),('raise','ValueError','invalid table collision surface')),
                            (surface(),('raise','RuntimeError','transform visited'))]:
        old=run('table',source,BadTransform(),monkeypatch,cache=False)
        new=run('table',source,BadTransform(),monkeypatch,cache=True)
        assert old[0]==new[0]==expected
        assert not new[2]._local_bounds._entries


def test_table_custom_array_conversion_remains_uncached(monkeypatch):
    class Custom:
        def __array__(self,*args,**kwargs):return surface()
    old=run('table',Custom(),np.eye(4),monkeypatch,cache=False)
    new=run('table',Custom(),np.eye(4),monkeypatch,cache=True)
    assert old[:2]==new[:2] and not new[2]._local_bounds._entries


@pytest.mark.parametrize('kind',['table','bin'])
def test_warm_bounds_do_not_cache_transform_or_collision_verdict(kind,monkeypatch):
    calls=[]
    def predicate(corners,world,**kwargs):calls.append(signature(world));return True
    o=fixture(kind,cache=True,predicate=predicate,monkeypatch=monkeypatch)
    a=surface();near=np.eye(4);far=near.copy();far[0,3]=10.
    assert o.intersects(a,near,True)
    assert not o.intersects(a,far,True)
    shifted=near.copy();shifted[0,3]=.1
    assert o.intersects(a,shifted,True)
    assert len(o._local_bounds._entries)==1 and len(calls)==2
    assert calls[0]==signature(a) and calls[1]==signature(a+[.1,0.,0.])


def test_bin_custom_methods_preserve_call_order_and_remain_uncached(monkeypatch):
    class Custom:
        def __init__(self):self.trace=[]
        def min(self,*,axis):self.trace.append(('min',axis));return surface().min(axis=axis)
        def max(self,*,axis):self.trace.append(('max',axis));return surface().max(axis=axis)
        def __matmul__(self,other):self.trace.append('matmul');return surface()@other
    old_source,new_source=Custom(),Custom()
    old=run('bin',old_source,np.eye(4),monkeypatch,cache=False)
    new=run('bin',new_source,np.eye(4),monkeypatch,cache=True)
    assert old[:2]==new[:2] and old_source.trace==new_source.trace
    assert not new[2]._local_bounds._entries
