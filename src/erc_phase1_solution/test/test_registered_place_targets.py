"""Real computed geometry policy; no IK, controller or external scene fixture."""
from copy import deepcopy
import math
import numpy as np
import pytest

from erc_phase1_solution.book_centered_place import book_centered_place_target
from erc_phase1_solution.registered_place_targets import registered_place_proposals, _interpolate
from test_bin_scene_admission import fixture as scene_fixture
from test_book_centered_place import held_corners


def inputs(scene=None, corners=None):
    scene=scene_fixture()[0] if scene is None else scene
    corners=held_corners() if corners is None else corners
    target=book_centered_place_target(corners,scene['floor_center'],.27,bin_rotation=scene['rotation'])
    release=target.tool_position;above=release.copy();above[2]+=.23
    first=above.copy();first[0]=.60
    positions=[first,*_interpolate(first,above,.06),*_interpolate(above,release,.06)]
    return positions,target,corners,scene


def propose(values):
    positions,target,corners,scene=values
    return registered_place_proposals(positions,target.tool_rotation,corners,scene,step=.06)


def test_ordered_computed_shifts_retain_all_corners_and_original_worst_margin():
    values=inputs();positions,target,corners,scene=values
    proposals=propose(values)
    assert [p.diagnostics['offset_m'] for p in proposals]==pytest.approx([0.,.025,.05,.055])
    frame=np.asarray(scene['rotation']);origin=np.asarray(scene['origin'])
    for proposal in proposals:
        d=proposal.diagnostics
        assert d['selected_minimum_wall_reserve_m']>=d['original_minimum_wall_reserve_m']-1e-12
        assert d['effective_material_margin_m']==pytest.approx(scene['modeled_margin_m']+scene['registration_margin_m'])
        projected=(corners@proposal.target.tool_rotation.T+proposal.target.tool_position-origin)@frame
        bounds=np.asarray(d['shrunken_cavity_horizontal_bounds'])
        assert np.all(projected[:,[0,2]]>=bounds[0]-1e-12)
        assert np.all(projected[:,[0,2]]<=bounds[1]+1e-12)
        np.testing.assert_allclose(projected[:,[0,2]],d['projected_book_corners'],atol=1e-12)
        np.testing.assert_array_equal(proposal.target.tool_rotation,target.tool_rotation)
        assert ((proposal.target.book_center-scene['floor_center'])@frame[:,1])==pytest.approx(.27)
        assert proposal.positions[0][0]==.60
        assert proposal.positions[0][2]-proposal.positions[-1][2]==pytest.approx(.23)
        # Every original proposal segment endpoint survives exact midpoint insertion.
        coarse=proposal.positions[::2]
        assert len(coarse)==d['coarse_cartesian_waypoints']
        for i,(a,b) in enumerate(zip(coarse,coarse[1:])):
            np.testing.assert_array_equal(proposal.positions[2*i+1],(a+b)/2.)
        assert all(not p.flags.writeable for p in proposal.positions)


def test_near_side_direction_comes_from_measured_cad_axis_not_world_x():
    scene=scene_fixture()[0];frame=np.asarray(scene['rotation'])
    frame[:,[0,2]]*=-1
    scene['rotation']=frame.tolist()
    scene['floor_center']=(np.asarray(scene['origin'])+frame@np.array([0.,-.095,0.])).tolist()
    proposals=propose(inputs(scene))
    shift=np.asarray(proposals[-1].diagnostics['displacement'])
    assert shift@frame[:,2]>0.
    center=proposals[0].target.book_center
    assert np.linalg.norm((center+shift)[:2])<np.linalg.norm(center[:2])


def test_padding_is_consumed_once_and_tight_depth_slack_clips_the_grid():
    corners=held_corners();center=corners.mean(axis=0)
    # Increase the long edge to leave only10mm surplus beyond tightest wall.
    axis=(corners[1]-corners[0]);axis/=np.linalg.norm(axis)
    corners=corners+((corners-center)@axis)[:,None]*axis[None,:]*(.37/.28-1.)
    proposals=propose(inputs(corners=corners))
    assert [p.diagnostics['offset_m'] for p in proposals]==pytest.approx([0.,.01])


def test_outputs_do_not_alias_caller_pose_or_scene_inputs():
    values=inputs();proposals=propose(values)
    before=[np.asarray(p.positions).copy() for p in proposals]
    values[0][0][:]=9.;values[2][:]=8.;values[3]['origin'][0]=7.
    for proposal,expected in zip(proposals,before):np.testing.assert_array_equal(proposal.positions,expected)


@pytest.mark.parametrize('fault',['margin','path','rotation','corner_order','scene','ambiguous'])
def test_invalid_geometry_is_rejected_without_an_unchecked_target(fault):
    positions,target,corners,scene=inputs()
    if fault=='margin':scene['modeled_margin_m']=.10
    if fault=='path':positions[1]=np.asarray(positions[1])+[0.,.01,0.]
    if fault=='rotation':target.tool_rotation[0,0]+=.1
    if fault=='corner_order':corners[[3,7]]=corners[[7,3]]
    if fault=='scene':scene['valid']=False
    if fault=='ambiguous':
        frame=np.asarray(scene['rotation']);floor=np.asarray(scene['floor_center'])
        floor[:2]-=frame[:2,2]*(floor[:2]@frame[:2,2])
        scene['origin']=(floor-frame@np.array([0.,-.095,0.])).tolist();scene['floor_center']=floor.tolist()
        positions,target,corners,scene=inputs(scene)
    with pytest.raises((ValueError,RuntimeError)):
        registered_place_proposals(positions,target.tool_rotation,corners,scene,step=.06)
