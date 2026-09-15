"""Pure planning regressions; no ROS imports or controller commands."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution import lift_first_extraction as lift
from erc_phase1_solution.kinematics import URDFChain


class CartesianChain:
    active_names = ('torso_lift_joint', 'x', 'y', 'z', 'a', 'b', 'c', 'd')
    lower = np.asarray([0., -4., -4., -4., -4., -4., -4., -4.])
    upper = np.asarray([.35, 4., 4., 4., 4., 4., 4., 4.])
    pose_error = staticmethod(URDFChain.pose_error)

    def __init__(self):
        self.targets = []

    def forward(self, q):
        result = np.eye(4)
        result[:3, 3] = np.asarray(q)[1:4]
        return result

    def solve(self, target, seeds, **kwargs):
        self.targets.append((target.copy(), [q.copy() for q in seeds], dict(kwargs)))
        q = seeds[0].copy()
        q[1:4] = target[:3, 3]
        return q, 0.


@pytest.fixture
def scenario(monkeypatch):
    chain = CartesianChain()
    front = np.asarray([.675, 0., 1.515])
    grasp = np.asarray([.35, .700, 0., 1.500, 0., 0., 0., 0.])
    goals = [grasp.copy(), grasp.copy()]
    goals[0][1], goals[1][1] = .660, .620
    triangle = np.asarray([[[0., -.02, -.02], [0., .02, -.02], [.02, 0., .02]]])
    meshes = {name: triangle.copy() for name in (lift.PALM_COLLISION_LINK, *lift.FINGER_COLLISION_LINKS)}
    calls = []
    def corners(point, q):
        signs = np.asarray([(x, y, z) for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
        return point + [.08, 0, 0] + signs*np.asarray([.095, .025, .140]) - chain.forward(q)[:3, 3]
    node = SimpleNamespace(chain=chain, carried_book_dimensions=np.asarray([.16,.02,.25]),
        carried_transition_samples=61, _attached_book_corners=corners,
        _shelf_cradle_geometry=SimpleNamespace(local_surfaces=lambda *args: meshes),
        _carried_robot_transition_is_safe=lambda start, end, book:
            calls.append(('robot', start.copy(), end.copy(), book.copy())) or True)
    def tool(node, front, grasp, start, end, plane, **kwargs):
        calls.append(('tool', start.copy(), end.copy(), plane, kwargs))
        return None
    monkeypatch.setattr(lift, 'check_cradle_tool_sweep', tool)
    bay = lift.RelativeShelfBay([.61,0.,2.26], [1.,0.,0.], 3,
        lateral_uncertainty_m=.030, roof_uncertainty_m=.010,
        assume_upright_supported=True, source='test onboard marker data')
    return node, front, grasp, goals, bay, calls, meshes


def plan(scenario, **changes):
    node, front, grasp, goals, bay, *_ = scenario
    return lift.plan_lift_first_extraction(node, front, grasp, goals,
        bay=changes.pop('bay', bay), aperture=changes.pop('aperture', .0183), **changes)


def test_lift_precedes_raised_withdrawal_without_mutating_approach(scenario):
    node, front, grasp, goals, _, calls, _ = scenario
    originals = [x.copy() for x in [front, grasp, *goals]]
    result = plan(scenario)
    assert len(result.route) == 3
    assert result.route[0][1:4] == pytest.approx([.700,0.,1.505])
    assert [q[1] for q in result.route] == pytest.approx([.700,.660,.620])
    assert [q[3] for q in result.route] == pytest.approx([1.505]*3)
    assert all(np.array_equal(old,new) for old,new in zip(originals,[front,grasp,*goals]))
    assert all(np.array_equal(t[0][:3,:3],np.eye(3)) for t in node.chain.targets)
    assert np.array_equal(node.chain.targets[0][1][0],grasp)
    assert all(t[2]['fixed_positions'] == {'torso_lift_joint': .35} for t in node.chain.targets)
    assert [c[0] for c in calls] == ['robot','tool']*3
    assert np.array_equal(calls[0][1],grasp)
    assert np.array_equal(calls[2][1],result.route[0])
    assert all(c[3] is None and c[4]['aperture']==.0183 for c in calls if c[0]=='tool')
    assert all(r['samples']>=61 for r in result.metrics['legs'])
    assert result.metrics['full_shelf_collision_certificate'] is False
    assert result.metrics['deferred_carry_regenerated'] is False
    assert result.metrics['legs'][0]['minimum_corner_rise_m'] == pytest.approx(0.)
    assert result.metrics['legs'][1]['minimum_corner_rise_m'] == pytest.approx(.005)
    result.terminal[1] = 99.
    assert result.route[-1][1] == pytest.approx(.620)


@pytest.mark.parametrize('change', [
    {'assume_upright_supported':False}, {'source':''}, {'physical_column':0},
    {'physical_column':6}, {'physical_column':True}, {'physical_column':2.5},
    {'marker_center_base':[0,0,np.nan]}, {'inward_axis_base':[2,0,0]},
    {'inward_axis_base':[1,0,.1]}, {'lateral_uncertainty_m':np.nan},
    {'roof_uncertainty_m':-.1}, {'margin_m':.014},
    {'lateral_uncertainty_m':.50},
])
def test_missing_or_invalid_bay_is_rejected_before_ik(scenario, change):
    with pytest.raises(ValueError):
        plan(scenario,bay=replace(scenario[4],**change))
    assert not scenario[0].chain.targets


@pytest.mark.parametrize('value', [0.,-.001,.021,np.nan,.0001])
def test_rise_is_bounded_and_greater_than_endpoint_tolerance(scenario,value):
    with pytest.raises(ValueError): plan(scenario,lift_m=value)


def test_uncertain_roof_rejected_including_known_book_height(scenario):
    with pytest.raises(ValueError,match='roof clearance'):
        plan(scenario,bay=replace(scenario[4],roof_uncertainty_m=.040))


def test_side_guard_uses_sensor_anchor_and_uncertainty(scenario):
    with pytest.raises(ValueError,match='side clearance'):
        plan(scenario,bay=replace(scenario[4],marker_center_base=[.61,.48,2.26]))


def test_side_guard_projects_on_registered_shelf_axis(scenario):
    # A 90-degree registered shelf means base x, not base y, is lateral.
    with pytest.raises(ValueError,match='side clearance'):
        plan(scenario,bay=replace(scenario[4],marker_center_base=[0,0,2.26],
            inward_axis_base=[0,1,0]))


@pytest.mark.parametrize('which',['robot','tool'])
def test_existing_collision_rejections_propagate(scenario,monkeypatch,which):
    if which=='robot': scenario[0]._carried_robot_transition_is_safe=lambda *args:False
    else: monkeypatch.setattr(lift,'check_cradle_tool_sweep',lambda *args,**kwargs:'palm_intersection')
    with pytest.raises(ValueError,match='sweep rejected'): plan(scenario)


def test_every_interior_sample_can_reject_a_downward_sag(scenario):
    chain=scenario[0].chain
    original=chain.forward
    def sag(q):
        result=original(q)
        # First vertical leg interior: endpoints still solve exactly.
        if 1.501 < q[3] < 1.504: result[2,3]-=.020
        return result
    chain.forward=sag
    with pytest.raises(ValueError,match='book-corner descent'): plan(scenario)


def test_raised_leg_interior_roof_bulge_is_rejected(scenario):
    chain=scenario[0].chain;original=chain.forward
    def bulge(q):
        result=original(q)
        if .67 < q[1] < .69:result[2,3]+=.05
        return result
    chain.forward=bulge
    with pytest.raises(ValueError,match='roof clearance'):plan(scenario)


def test_gripper_roof_geometry_is_checked_not_only_book(scenario):
    scenario[6][lift.PALM_COLLISION_LINK][:,:,2]+=.3
    with pytest.raises(ValueError,match='roof clearance'):plan(scenario)


def test_incomplete_gripper_geometry_fails_closed(scenario):
    del scenario[6][lift.FINGER_COLLISION_LINKS[-1]]
    with pytest.raises(ValueError,match='incomplete'):plan(scenario)


def test_nonfinite_and_legacy_book_dimensions_reject(scenario):
    scenario[0].carried_book_dimensions[1]=.03
    with pytest.raises(ValueError,match='20 mm'):plan(scenario)


def test_invalid_ik_results_do_not_reach_sweeps(scenario):
    scenario[0].chain.solve=lambda *args,**kwargs:(np.full(8,np.nan),0.)
    with pytest.raises(ValueError,match='finite'):plan(scenario)
    assert not scenario[5]


def test_bad_rigid_transform_rejected(scenario):
    original=scenario[0].chain.forward
    def broken(q):
        result=original(q);result[0,0]=2.;return result
    scenario[0].chain.forward=broken
    with pytest.raises(ValueError,match='rigid transform'):plan(scenario)


def test_live_node_and_explicit_finger_feedback_reach_guards(scenario,monkeypatch):
    node=scenario[0];seen=[]
    fingers={'gripper_left_finger_joint':.01831}
    monkeypatch.setattr(lift,'check_cradle_tool_sweep',
        lambda current,*args,**kwargs:seen.append((current,kwargs['finger_positions'])) or None)
    plan(scenario,finger_positions=fingers)
    assert len(seen)==3 and all(n is node and f is fingers for n,f in seen)


def test_pitched_grasp_preserves_initial_upright_book_transform(scenario):
    chain=scenario[0].chain; original=chain.forward
    angle=-.5
    rotation=np.asarray([[np.cos(angle),0.,-np.sin(angle)],
                         [0.,-1.,0.],[-np.sin(angle),0.,-np.cos(angle)]])
    def pitched(q):
        pose=original(q);pose[:3,:3]=rotation;return pose
    chain.forward=pitched
    result=plan(scenario)
    assert all(np.array_equal(target[0][:3,:3],rotation) for target in chain.targets)
    assert result.metrics['legs'][0]['minimum_corner_rise_m']==pytest.approx(0.)
    assert result.metrics['legs'][1]['minimum_corner_rise_m']==pytest.approx(.005)


def test_failed_later_ik_does_not_return_partial_route_or_mutate_inputs(scenario):
    chain=scenario[0].chain;original=chain.solve
    before=[q.copy() for q in scenario[3]]
    def failed(target,seeds,**kwargs):
        if target[0,3] < .69:return None,float('inf')
        return original(target,seeds,**kwargs)
    chain.solve=failed
    with pytest.raises(ValueError,match='IK failed at leg 1'):plan(scenario)
    assert all(np.array_equal(a,b) for a,b in zip(before,scenario[3]))


def test_exact_route_recheck_never_resolves_ik_or_mutates_cache(scenario):
    node,front,grasp,_,bay,_,_=scenario
    result=plan(scenario)
    count=len(node.chain.targets)
    cache=object();node._cached_post_retreat_plan=cache
    original=[q.copy() for q in result.route]
    fingers={'gripper_left_finger_joint':.01831}
    measured=grasp.copy();measured[1]+=.0001
    checked=lift.validate_lift_first_route(node,front,measured,result.route,
        bay=bay,aperture=.01831,finger_positions=fingers,
        attached_corners=result.attached_corners)
    assert len(node.chain.targets)==count
    assert node._cached_post_retreat_plan is cache
    assert all(np.array_equal(x,y) for x,y in zip(original,result.route))
    assert checked['supplied_finger_positions_checked']
    assert checked['route_replanned'] is False
    assert len(checked['legs'])==3


def test_exact_route_rechecks_measured_start_to_first_lift_not_planned_start(scenario):
    node,front,grasp,_,bay,_,_=scenario
    result=plan(scenario)
    measured=grasp.copy();measured[3]+=.006
    with pytest.raises(ValueError,match='book-corner descent'):
        lift.validate_lift_first_route(node,front,measured,result.route,
            bay=bay,aperture=.0183)


def test_exact_route_rechecks_actual_passive_tool_bay_clearance(scenario):
    node,front,grasp,_,bay,_,meshes=scenario
    result=plan(scenario)
    meshes[lift.FINGER_COLLISION_LINKS[0]][:,:,2]+=.3
    with pytest.raises(ValueError,match='uncertain bay clearance'):
        lift.validate_lift_first_route(node,front,grasp,result.route,
            bay=bay,aperture=.0183,finger_positions={'passive':.2})


def test_exact_route_rechecks_every_withdrawal_interior(scenario):
    node,front,grasp,_,bay,_,_=scenario
    result=plan(scenario);original=node.chain.forward
    def sag(q):
        pose=original(q)
        if .63 < q[1] < .65:pose[2,3]-=.02
        return pose
    node.chain.forward=sag
    with pytest.raises(ValueError,match='book-corner descent at leg 2'):
        lift.validate_lift_first_route(node,front,grasp,result.route,
            bay=bay,aperture=.0183)


@pytest.mark.parametrize('checked',[False,True])
def test_modeled_tool_allowance_preserves_book_clearance_margins(scenario,checked):
    node,front,grasp,_,bay,_,_=scenario
    original=plan(scenario)
    if checked:
        normal=lift.validate_lift_first_route(node,front,grasp,original.route,bay=bay,aperture=.0183)
        padded=lift.validate_lift_first_route(node,front,grasp,original.route,bay=bay,aperture=.0183,modeled_tool_allowance_m=.005)
    else:
        normal=original.metrics;padded=plan(scenario,modeled_tool_allowance_m=.005).metrics
    # The taller book controls roof clearance; no extra5 mm is subtracted from it.
    for a,b in zip(normal['legs'],padded['legs']):
        assert a['roof_clearance_after_margin_uncertainty_m']==pytest.approx(b['roof_clearance_after_margin_uncertainty_m'])
        assert a['minimum_corner_rise_m']==pytest.approx(b['minimum_corner_rise_m'])
    assert padded['modeled_tool_allowance_m']==.005
    assert padded['modeled_tool_allowance_is_certified_deflection_bound'] is False


@pytest.mark.parametrize('checked',[False,True])
@pytest.mark.parametrize('edge',['roof','side'])
def test_modeled_tool_allowance_rejects_near_roof_and_side(scenario,checked,edge):
    node,front,grasp,_,bay,_,meshes=scenario
    if edge=='roof':meshes[lift.FINGER_COLLISION_LINKS[0]][:,:,2]+=.136
    else:meshes[lift.FINGER_COLLISION_LINKS[0]][:,:,1]+=.432
    original=plan(scenario)
    with pytest.raises(ValueError,match='clearance'):
        if checked:
            lift.validate_lift_first_route(node,front,grasp,original.route,bay=bay,aperture=.0183,modeled_tool_allowance_m=.005)
        else:plan(scenario,modeled_tool_allowance_m=.005)


@pytest.mark.parametrize('allowance',[-.001,.006,float('nan')])
def test_invalid_modeled_tool_allowance_rejected(scenario,allowance):
    with pytest.raises(ValueError,match='modeled tool allowance'):
        plan(scenario,modeled_tool_allowance_m=allowance)
