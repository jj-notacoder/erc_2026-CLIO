"""Small sufficient-separation and actual sample-wiring controls; no ROS/assets.

Prepared without execution. Narrow-phase spies deliberately report collision:
only independently separated tiny fixtures may bypass them. They do not certify
the full robot or promise equivalence with conservative legacy degeneracies.
"""
import ast
import copy
import importlib.util
import itertools
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
# Exact production types let custom fixtures retain the original fallback.
from erc_phase1_solution.scene_checked_place import (
    NominalBinObstacle, TableSceneObstacle, ScreenEnvelope, ShelfCradleGeometry,
)
import pytest

PACKAGE = Path(__file__).resolve().parents[1] / 'erc_phase1_solution'
spec = importlib.util.spec_from_file_location('moving_axis_candidate', PACKAGE/'static_pair_separation.py')
separator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(separator)
separated = separator.separated_on_axes


def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def box(center=(0.,0.,0.), size=(1.,1.,1.)):
    corners=np.array(list(itertools.product((-1.,1.), repeat=3)))*np.asarray(size)/2+center
    indices=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],
                      [0,4,5],[0,5,1],[2,3,7],[2,7,6],
                      [0,2,6],[0,6,4],[1,5,7],[1,7,3]])
    return corners[indices]


@pytest.mark.parametrize('angle', [0.,.37,1.11])
@pytest.mark.parametrize('gap,expected', [(-.001,False),(0.,False),(99e-6,False),(100e-6,False),(101e-6,True),(.2,True)])
def test_rotated_touch_overlap_and_strict_margin(angle,gap,expected):
    r=rotation(angle)
    a=box()@r.T; b=box((1.+gap,0.,0.))@r.T
    assert separated(a,b,r.T) is expected


def test_complete_compound_and_containment_cannot_clear():
    a=box()
    assert not separated(box(size=(3.,3.,3.)),a,np.eye(3))
    compound=np.concatenate([box((10.,0.,0.)),box((.1,0.,0.))])
    assert not separated(a,compound,np.eye(3))


def test_no_stale_verdict_or_mutation_of_inputs():
    a,b=box(),box((2.,0.,0.)); axes=np.eye(3)
    saved=[x.copy() for x in (a,b,axes)]
    assert separated(a,b,axes)
    for before,after in zip(saved,(a,b,axes)): np.testing.assert_array_equal(before,after)
    b[...,0]-=2.
    assert not separated(a,b,axes)
    b[...,0]+=2.
    assert separated(a,b,axes)


@pytest.mark.parametrize('axes',[[],np.zeros((1,3)),[[np.nan,0,0]],[[np.inf,0,0]],[[1,2]],np.ones((7,3)),np.ones((1,1,3))])
def test_uncertain_axes_fall_through(axes):
    assert not separated(box(),box((2.,0.,0.)),axes)


@pytest.mark.parametrize('surface',[np.empty((0,3,3)),np.zeros((3,3)),np.full((1,3,3),np.inf),np.full((1,3,3),np.nan)])
def test_invalid_mesh_falls_through(surface):
    assert not separated(box(),surface,np.eye(3))


@pytest.mark.parametrize('scale',[1e-300,1e300,-1e300])
def test_axes_normalize_without_scale_dependent_clearance(scale):
    assert separated(box(),box((2.,0.,0.)),np.eye(3)*scale)
    assert not separated(box(),box((1.,0.,0.)),np.eye(3)*scale)


def test_large_operands_and_overflow_are_conservative():
    r=rotation(math.pi/4)
    a=box(size=(2.,.01,.01))@r.T+np.array([1e12,1e12,0.])
    assert not separated(a,a+r[:,1]*.02,r.T)
    huge=np.full((1,3,3),1e308)
    assert not separated(huge,-huge,np.eye(3))


def test_stale_axes_are_only_hints_and_degenerate_touch_is_not_clear():
    a=np.array([[[0.,0.,0.],[1.,1.,0.],[2.,2.,0.]]])
    axis=np.array([[1.,-1.,0.]])
    assert separated(a,a+[0.,.2,0.],axis)
    assert not separated(a,a,axis)
    assert not separated(a,a+[2.,2.,0.],axis)
    # An irrelevant hint can miss a separation; it cannot authorize overlap.
    assert not separated(box(),box((2.,0.,0.)),[[0.,1.,0.]])


class Fixture:
    def __init__(self, *, fail=None, compound=False):
        self.trace=[]; self.fail=fail; self.right=np.zeros(7); self.right[0]=.2
        self.head=np.zeros(2); self.cancelled=False
        r=rotation(.6); self.local=box(size=(2.,.1,.1))
        robot=self.local@r.T+[0.,0.,1.]
        if compound: robot=np.concatenate([robot,robot-r[:,1]*.2])
        self.meshes=[NS(link='arm_left_3_link',triangles=robot,watertight=True)]
        self.node=NS(_cancel=NS(is_set=lambda:self.cancelled),_lock=None,
            _resolved_right_positions=lambda _:self.right.copy(),
            _resolved_head_positions=lambda _:self.head.copy(),
            _collision_link_transforms=self.transforms,chain=NS(forward=self.hand),
            carried_collision_meshes=self.meshes,_robot_self_collision=self.body,
            _watertight_collision_links=lambda:{'arm_left_3_link'})
        self.obstacle=NS(intersects=lambda *args:self.record('obstacle'))
        self.tool=NS(local_surfaces=lambda aperture:{'tip':self.local+np.array([0.,aperture-.017,0.])},
                     watertight={'tip':True})
        self.q=np.array([.35,.6,0.,0.,0.,0.,0.,0.])

    def record(self,label):
        self.trace.append(label)
        return self.fail==label

    def transforms(self,q,**context):
        t=np.eye(4); t[:3,3]=rotation(.6)[:,1]*(context['right_positions'][0]+context['head_positions'][0])
        return {'arm_left_3_link':t}

    def hand(self,q):
        t=np.eye(4); t[:3,:3]=rotation(q[1]); t[:3,3]=[0.,0.,1.]
        t[:3,3]+=t[:3,1]*q[2]
        return t

    def body(self,q,**context):
        return ('a','b') if self.record('body') else None

    def prepare(self,surface):
        self.trace.append('prepare'); return surface

    def narrow(self,*args,**kwargs):
        self.trace.append('narrow'); return True

    def certificate(self,*args):
        self.trace.append('certificate'); return separated(*args)

    def checker(self):
        source=PACKAGE/'scene_checked_place.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        names={'_finite','_SampleWorldAabbs','PlaceSceneChecker'}
        body=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        assert {n.name for n in body}==names
        namespace=dict(np=np,math=math,PALM_COLLISION_LINK='gripper_left_base_link',
            PreparedTriangleMesh=self.prepare,triangle_meshes_intersect=self.narrow,
            separated_on_axes=self.certificate,
            NominalBinObstacle=NominalBinObstacle, TableSceneObstacle=TableSceneObstacle,
            ScreenEnvelope=ScreenEnvelope, ShelfCradleGeometry=ShelfCradleGeometry)
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*body],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
        return namespace['PlaceSceneChecker'](self.node,self.obstacle,self.tool,np.zeros((8,3)))


def test_actual_sample_current_rotated_meshes_skip_only_narrow_work():
    f=Fixture(); c=f.checker()
    assert c.sample(f.q,.017,False)
    assert f.trace==['obstacle','obstacle','body','certificate']


@pytest.mark.parametrize('change',['right','head','q','aperture'])
def test_actual_sample_changed_context_reprojects_before_accepting(change):
    f=Fixture(); c=f.checker(); assert c.sample(f.q,.017,False)
    aperture=.017
    if change=='right': f.right[0]=0.
    elif change=='head': f.head[0]=-.2
    elif change=='q': f.q[2]=.2
    else: aperture=.217  # Pure geometry fixture, not a controller command.
    assert not c.sample(f.q,aperture,False)
    assert c.last_rejection=={'reason':'tool_robot','tool_link':'tip','robot_link':'arm_left_3_link'}
    assert f.trace[-4:]==['certificate','prepare','prepare','narrow']


def test_actual_sample_compound_collision_and_earlier_rejection_preserved():
    f=Fixture(compound=True); c=f.checker()
    assert not c.sample(f.q,.017,False)
    assert c.last_rejection['reason']=='tool_robot'
    for fail,reason in [('obstacle','robot_bin'),('body','robot_self')]:
        f=Fixture(fail=fail); c=f.checker()
        assert not c.sample(f.q,.017,False)
        assert c.last_rejection['reason']==reason
        assert 'certificate' not in f.trace


def test_actual_sample_cancel_before_geometry_and_aabb_before_certificate():
    f=Fixture(); c=f.checker(); f.cancelled=True
    assert not c.sample(f.q,.017,False)
    assert c.last_rejection=={'reason':'cancelled'} and not f.trace
    f=Fixture(); f.right[0]=5.; c=f.checker()
    assert c.sample(f.q,.017,False)
    assert 'certificate' not in f.trace and 'narrow' not in f.trace
