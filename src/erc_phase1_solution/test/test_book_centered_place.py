import itertools
import math
import numpy as np
import pytest

from erc_phase1_solution.book_centered_place import book_centered_place_target


def held_corners(padding=0.015):
    c,s=math.cos(.5),math.sin(.5)
    grasp_rotation=np.asarray([[c,0.,s],[0.,-1.,0.],[s,0.,-c]])
    half=np.asarray([.16,.02,.25])/2+padding
    signs=np.asarray(list(itertools.product((-1.,1.),repeat=3)))
    return np.asarray([.055,-.001,.013])+(signs*half)@grasp_rotation


@pytest.mark.parametrize('point',([.868,.004,.751],[.8,.3,.74]))
def test_transformed_book_is_centered_with_long_axis_along_observed_ray(point):
    corners=held_corners();point=np.asarray(point)
    target=book_centered_place_target(corners,point,.27)
    placed=corners@target.tool_rotation.T+target.tool_position
    np.testing.assert_allclose(placed.mean(axis=0),point+[0.,0.,.27],atol=1e-12)
    along=np.r_[point[:2],0.]/np.linalg.norm(point[:2])
    long_edge=placed[1]-placed[0]
    np.testing.assert_allclose(long_edge/np.linalg.norm(long_edge),along,atol=1e-12)
    thickness=placed[2]-placed[0]
    np.testing.assert_allclose(thickness/np.linalg.norm(thickness),[0.,0.,-1.],atol=1e-12)
    assert target.tool_rotation[2,1] > .999999
    assert np.linalg.det(target.tool_rotation)==pytest.approx(1.)


def test_existing_padding_does_not_shift_target_or_add_padding_again():
    plain=book_centered_place_target(held_corners(0.),[.8,.02,.75],.27)
    padded=book_centered_place_target(held_corners(.015),[.8,.02,.75],.27)
    np.testing.assert_allclose(plain.tool_position,padded.tool_position,atol=1e-12)
    np.testing.assert_allclose(plain.tool_rotation,padded.tool_rotation,atol=1e-12)


def test_rejects_inconsistent_corner_order_before_producing_target():
    corners=held_corners();corners[[3,7]]=corners[[7,3]]
    with pytest.raises(ValueError):book_centered_place_target(corners,[.8,0.,.75],.27)


@pytest.mark.parametrize('height',(0.,-1.,float('nan')))
def test_rejects_invalid_height(height):
    with pytest.raises(ValueError):book_centered_place_target(held_corners(),[.8,0.,.75],height)


def test_rejects_degenerate_box_and_missing_approach_direction():
    with pytest.raises(ValueError):book_centered_place_target(np.zeros((8,3)),[.8,0.,.75],.27)
    with pytest.raises(ValueError):book_centered_place_target(held_corners(),[0.,0.,.75],.27)


class StopAtCartesianPlanning(RuntimeError):
    pass


def place_fixture(enabled):
    from erc_phase1_solution.manipulation_node import ManipulationNode
    node=object.__new__(ManipulationNode)
    node._held_book_corners=held_corners()
    node._carried_staging_solution=np.asarray([.35,.36,-1.83,.47,-2.35,0.,-1.2,0.])
    node._gravity_supported_payload=True
    node._fresh_retention_probe=lambda *args,**kwargs:True
    node._measured_left_solution=lambda:node._carried_staging_solution.copy()
    node._wait_for_perception_point=lambda name:np.asarray([.868,.004,.751])
    node.place_clearance=.22;node.cartesian_clearance=.45;node.cartesian_step=.06
    node.place_torso_height=.30;node.place_joint_limit_margin=.03
    node.book_centered_place_enabled=enabled
    node.book_centered_place_height_above_point=.27
    node.book_centered_place_approach_x=.60
    node._supported_compact_goals=lambda q:[q.copy()]
    checked=[]
    node._carried_robot_transition_is_safe=lambda *args,**kwargs:checked.append('robot') or True
    node._gravity_supported_transition_is_safe=lambda *args,**kwargs:checked.append('support') or True
    node._plan_carried_joint_route=lambda start,goals,*args,**kwargs:checked.append('staging_route') or list(goals)
    captured={}
    def capture(positions,rotations,torso,**kwargs):
        captured.update(positions=np.asarray(positions),rotations=rotations,kwargs=kwargs)
        raise StopAtCartesianPlanning()
    node._solve_cartesian_path=capture
    return node,captured,checked


@pytest.mark.parametrize('enabled',(False,True))
def test_place_wires_rotation_specific_tcp_and_keeps_existing_checks(enabled, monkeypatch):
    node,captured,checked=place_fixture(enabled)
    if enabled:
        def scene_plan(node, positions, rotation, carried, torso, seed, point, resolver, *, table_scene=None):
            captured.update(positions=np.asarray(positions), rotations=(rotation,))
            assert table_scene is None  # This target-wiring fixture does not enable table sensing.
            np.testing.assert_array_equal(carried, node._measured_left_solution())
            assert torso[0] == .30 and seed[0] == .30
            raise StopAtCartesianPlanning()
        monkeypatch.setattr('erc_phase1_solution.manipulation_node.plan_scene_checked_place', scene_plan)
    with pytest.raises(StopAtCartesianPlanning):node._place()
    if enabled:
        # New scene helper owns validation from actual carry; no reverse staging.
        assert checked==[]
    else:
        assert checked==['robot','support','staging_route']
        assert callable(captured['kwargs']['candidate_validator'])
        assert captured['kwargs']['joint_limit_margin']==.10
        assert captured['kwargs'].get('transition_edge_validator') is None
    positions=captured['positions'];rotations=captured['rotations']
    if enabled:
        assert len(rotations)==1
        placed_center=rotations[0]@node._held_book_corners.mean(axis=0)+positions[-1]
        np.testing.assert_allclose(placed_center,[.868,.004,1.021],atol=1e-12)
        assert positions[0,0]==.60
    else:
        assert len(rotations)==3
        np.testing.assert_allclose(positions[-1],[1.038,.004,.971],atol=1e-12)
        assert positions[0,0]==.45


def test_failed_retention_prevents_new_target_path_from_running():
    node,captured,checked=place_fixture(True)
    node._fresh_retention_probe=lambda *args,**kwargs:False
    with pytest.raises(RuntimeError,match='not retained'):node._place()
    assert not captured and not checked


def test_supported_edge_filter_selects_an_alternate_existing_subset_route():
    from erc_phase1_solution.manipulation_node import ManipulationNode
    node=object.__new__(ManipulationNode)
    start=np.zeros(8);goal=np.r_[0.,np.ones(7)]
    def retracted(first,last):
        first=np.asarray(first);last=np.asarray(last)
        return bool(np.all(np.isin(first[1:],[0.,1.])) and np.all(np.isin(last[1:],[0.,1.]))
                    and np.count_nonzero(first[1:]!=last[1:])<=1)
    node._retracted_transition_is_safe=retracted
    original=node._plan_retracted_transition(start,goal)
    assert original is not None and original[0][1]==1. and original[0][2]==0.
    def supported(first,last):
        return not (last[1]==1. and last[2]==0.)
    selected=node._plan_retracted_transition(start,goal,edge_validator=supported)
    assert selected is not None and selected[0][1]==0. and selected[0][2]==1.
    previous=start
    for waypoint in [*selected,goal]:
        assert supported(previous,waypoint) and retracted(previous,waypoint)
        previous=waypoint


def test_edge_filter_rejection_cannot_be_bypassed_by_direct_or_subset_paths():
    from erc_phase1_solution.manipulation_node import ManipulationNode
    node=object.__new__(ManipulationNode)
    calls=[]
    node._retracted_transition_is_safe=lambda *args:calls.append(args) or True
    route=node._plan_retracted_transition(np.zeros(8),np.ones(8),edge_validator=lambda *args:False)
    assert route is None and calls==[]
