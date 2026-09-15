"""Registered planes and complete clipped triangles, independent of simulator poses."""
from dataclasses import replace
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution.empty_shelf_bounds import EmptyShelfBounds, _forward_vertices
from erc_phase1_solution.lift_first_extraction import RelativeShelfBay
from erc_phase1_solution.rigid_palm_preflight import LEFT_GRIPPER_COLLISION_LINKS


def fixture(*, angle=0., coordinates=None, dimensions=(.16,.02,.25), uncertainty=.010, bay_changes=None, tool=False):
    inward=np.array([np.cos(angle),np.sin(angle),0.])
    left=np.array([-inward[1],inward[0],0.])
    marker=np.array([.605,-.006,2.26])
    lip=marker+.001016*inward
    front=lip+.064*inward;front[2]=.604
    def world(local):
        local=np.asarray(local,dtype=float)
        out=lip+local[...,0,None]*inward+local[...,1,None]*left
        out[...,2]=local[...,2]
        return out
    if coordinates is None:coordinates=[[.06,-.01,.60],[.07,.01,.60],[.06,.01,.61]]
    surface=np.array([world(coordinates)])
    behind=np.array([world([[-.30,-.01,.60],[-.29,.01,.60],[-.30,.01,.61]])])
    locals={name:behind.copy() for name in LEFT_GRIPPER_COLLISION_LINKS}
    if tool:locals[next(iter(locals))]=surface
    node=NS(_cancel=threading.Event(),carried_transition_samples=61,
        carried_book_dimensions=np.array(dimensions),chain=NS(forward=lambda q:np.eye(4)),
        _world_collision_surfaces=lambda q,**kw:{'robot_test':behind if tool else surface})
    guard=NS(open_aperture=.069,geometry=NS(local_surfaces=lambda aperture:locals),
             context={'right_positions':np.zeros(7),'head_positions':np.zeros(2)})
    bay=RelativeShelfBay(marker_center_base=marker,inward_axis_base=inward,
        physical_column=3,lateral_uncertainty_m=.200,roof_uncertainty_m=.010,
        assume_upright_supported=True,source='independent registered unit fixture')
    if bay_changes:bay=replace(bay,**bay_changes)
    bounds=EmptyShelfBounds(node,front,None,guard,bay,normal_uncertainty_m=uncertainty)
    return NS(bounds=bounds,node=node,guard=guard,bay=bay,front=front,surface=surface)


@pytest.mark.parametrize('angle',[0.,.71,np.pi/2])
@pytest.mark.parametrize('coordinates,reason',[
    ([[.06,-.01,.60],[.07,.01,.60],[.06,.01,.61]],None),
    ([[.06,-.01,.48],[.07,.01,.48],[.06,.01,.49]],'floor'),
    ([[.06,-.01,.76],[.07,.01,.76],[.06,.01,.77]],'roof'),
    ([[.06,-.30,.60],[.07,-.29,.60],[.06,-.30,.61]],'side'),
    ([[.06,.30,.60],[.07,.29,.60],[.06,.30,.61]],'side'),
    ([[.28,-.01,.60],[.29,.01,.60],[.28,.01,.61]],'back'),
])
def test_rotated_registered_geometry_checks_all_bay_boundaries(angle,coordinates,reason):
    bounds=fixture(angle=angle,coordinates=coordinates).bounds
    result=bounds.sample(np.zeros(8),True)
    if reason is None:
        assert result is None
        assert min(bounds.minimum_floor,bounds.minimum_roof,bounds.minimum_side,bounds.minimum_back)>0.
    else:
        assert result.startswith('empty_bay_clearance:robot_test:')
        assert getattr(bounds,'minimum_'+reason)<0.


@pytest.mark.parametrize('angle',[0.,.37,1.8])
def test_clipping_is_rigid_yaw_translation_invariant_and_includes_crossings(angle):
    triangles=np.array([[[-1.,-2.,.5],[1.,2.,.7],[1.,0.,.6]]])
    c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    translation=np.array([3.,-2.,.4])
    expected=_forward_vertices(triangles,np.zeros(3),np.array([1.,0.,0.]),0.)
    actual=_forward_vertices(triangles@rotation.T+translation,translation,rotation[:,0],0.)
    restored=(actual-translation)@rotation
    assert restored.shape==expected.shape==(4,3)
    # The implementation retains original vertices then edge crossings; an
    # arbitrary rigid yaw must not change which boundary intersections exist.
    np.testing.assert_allclose(restored,expected,rtol=0.,atol=2e-15)
    assert np.all((actual-translation)@rotation[:,0]>=-1e-15)


def test_zero_plane_vertices_and_edges_are_retained_without_nonfinite_crossings():
    triangles=np.array([[[0.,0.,.6],[0.,1.,.6],[-1.,0.,.6]]])
    occupied=_forward_vertices(triangles,np.zeros(3),np.array([1.,0.,0.]),0.)
    assert occupied.shape==(2,3)
    assert np.isfinite(occupied).all()
    np.testing.assert_array_equal(occupied,triangles[0,:2])


@pytest.mark.parametrize('dimensions',[[.16,.02], [.16,.02,.25,.1], [.16,.03,.25], [.16,.02,-.25], [.16,.02,np.nan], [.16,.02,np.inf]])
def test_invalid_or_different_book_dimensions_reject_before_fk(dimensions):
    with pytest.raises(ValueError,match='official_book_dimensions'):fixture(dimensions=dimensions)


@pytest.mark.parametrize('uncertainty',[0.,.009,np.nextafter(.010,0.),.0501,np.nan,np.inf,-.01])
def test_normal_uncertainty_cannot_remove_or_exceed_admitted_reserve(uncertainty):
    with pytest.raises(ValueError,match='normal_uncertainty'):fixture(uncertainty=uncertainty)


@pytest.mark.parametrize('axis',[[0.,0.,0.],[2.,0.,0.],[0.,0.,1.],[np.nan,0.,0.]])
def test_invalid_registered_plane_normal_is_rejected(axis):
    with pytest.raises(ValueError):fixture(bay_changes={'inward_axis_base':axis})


@pytest.mark.parametrize('changes',[{'marker_center_base':[np.nan,0.,2.]},
    {'source':''},{'physical_column':True},{'physical_column':6},
    {'lateral_uncertainty_m':.5},{'roof_uncertainty_m':-.01},
    {'margin_m':.014},{'assume_upright_supported':False}])
def test_invalid_registration_or_uncertainty_is_rejected(changes):
    with pytest.raises(ValueError):fixture(bay_changes=changes)


@pytest.mark.parametrize('depth',[.014,.151,-.01])
def test_marker_book_depth_inconsistency_rejects(depth):
    f=fixture();front=f.bounds.registered_lip+depth*f.bounds.inward;front[2]=f.front[2]
    with pytest.raises(ValueError,match='marker_book_depth'):
        EmptyShelfBounds(f.node,front,None,f.guard,f.bay)


def test_tool_allowance_is_applied_to_floor_and_back_boundary():
    coordinates=[[.06,-.01,.498],[.07,.01,.498],[.06,.01,.499]]
    assert fixture(coordinates=coordinates).bounds.sample(np.zeros(8),True) is None
    assert fixture(coordinates=coordinates,tool=True).bounds.sample(np.zeros(8),True).startswith('empty_bay_clearance:')
    coordinates=[[.272,-.01,.60],[.273,.01,.60],[.272,.01,.61]]
    assert fixture(coordinates=coordinates).bounds.sample(np.zeros(8),True) is None
    assert fixture(coordinates=coordinates,tool=True).bounds.sample(np.zeros(8),True).startswith('empty_bay_clearance:')


def test_nonfinite_model_surface_rejects_before_clipping():
    f=fixture();f.surface[0,0,0]=np.nan
    assert f.bounds.sample(np.zeros(8),True)=='empty_shelf_nonfinite_surface:robot_test'


def test_cancelled_bounds_edge_does_not_evaluate_geometry():
    f=fixture();f.node._cancel.set()
    def forbidden(*args):raise AssertionError('cancelled bounds performed FK')
    f.node.chain.forward=forbidden
    with pytest.raises(RuntimeError,match='cancelled'):f.bounds.edge(np.zeros(8),np.ones(8),allow_entry=False)


def test_dense_edge_detects_collision_between_coarse_eighth_samples():
    f=fixture();seen=[]
    def sample(q,allow_entry):
        seen.append(float(q[1]))
        return 'narrow shelf crossing' if .049<q[1]<.051 else None
    f.bounds.sample=sample
    first=np.zeros(8);last=first.copy();last[1]=.3
    # Seven eighth-interval probes miss q1=.05; the full61 grid contains it.
    assert all(sample(first+(last-first)*x,False) is None for x in (.125,.25,.375,.5,.625,.75,.875))
    seen.clear()
    assert f.bounds.edge(first,last,allow_entry=False)=='narrow shelf crossing'
    assert any(.049<x<.051 for x in seen)
