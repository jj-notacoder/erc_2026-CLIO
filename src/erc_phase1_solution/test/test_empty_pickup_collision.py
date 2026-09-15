"""Portable tests of real geometry guards with small deliberate witnesses."""
import ast
from pathlib import Path
from types import SimpleNamespace
import threading
from collections import deque

import numpy as np
import pytest

from erc_phase1_solution.empty_pickup_collision import EmptyPickupCollision, measured_context, NAMES, MASTER
from erc_phase1_solution.kinematics import _box_triangles
from erc_phase1_solution.rigid_palm_preflight import LEFT_GRIPPER_COLLISION_LINKS, PALM_COLLISION_LINK
from erc_phase1_solution.motion_profiles import ARM_JOINTS, PREGRASP, HOME, OFFER


def box(center, size=.1):
    return _box_triangles([size]*3) + center


class Screen:
    def __init__(self, center=(10., 0., 1.)):
        self.center = np.array(center)
    def world(self, torso, head):
        return np.array([(x,y,z) for x in (-.05,.05) for y in (-.05,.05) for z in (-.05,.05)]) + self.center


def fixture():
    local = {link: box([20.+index,0.,1.]) for index,link in enumerate(LEFT_GRIPPER_COLLISION_LINKS)}
    geometry = SimpleNamespace(local_surfaces=lambda aperture: local,
        watertight={link: True for link in LEFT_GRIPPER_COLLISION_LINKS})
    node=SimpleNamespace(chain=SimpleNamespace(lower=np.full(8,-4.),upper=np.full(8,4.),forward=lambda q:np.eye(4)),
        gripper_open=.069,carried_transition_samples=61,_cancel=threading.Event(),
        _robot_self_collision=lambda q,**kw:None,_world_collision_surfaces=lambda q,**kw:{'arm_left_3_link':box([0.,0.,1.])},
        _watertight_collision_links=lambda:frozenset(['arm_left_3_link']),pick_torso_height=.35,
        adaptive_endpoint_tolerance=.0006)
    checker=EmptyPickupCollision(node,np.zeros(8),np.zeros(7),np.zeros(2),.069,geometry=geometry,screen=Screen())
    return checker,node,local


def test_clear_pose_and_exact_cache():
    c,n,_=fixture()
    assert c.sample(np.zeros(8),.069) is None
    assert c.sample(np.zeros(8),.069) is None
    assert (c.checked_samples,c.cache_hits)==(1,1)
    q=np.zeros(8); q[1]=np.nextafter(0.,1.)
    assert c.sample(q,.069) is None
    assert c.checked_samples==2


def test_cancel_precedes_cached_clearance():
    c,n,_=fixture();assert c.sample(np.zeros(8),.069) is None
    n._cancel.set(); assert c.sample(np.zeros(8),.069)=='empty_pickup_cancelled'


def test_existing_body_self_result_rejects():
    c,n,_=fixture();n._robot_self_collision=lambda q,**kw:('arm_left_3_link','head_2_link')
    assert 'robot_self' in c.sample(np.zeros(8),.069)


@pytest.mark.parametrize('robot_link',['arm_left_3_link','arm_right_2_link','torso_lift_link'])
def test_screen_box_rejects_robot_containment(robot_link):
    c,n,_=fixture();n._world_collision_surfaces=lambda q,**kw:{robot_link:box([10.,0.,1.])}
    assert c.sample(np.zeros(8),.069)==f'empty_pickup_screen:{robot_link}'


@pytest.mark.parametrize('tool_link',LEFT_GRIPPER_COLLISION_LINKS)
def test_every_open_tool_link_can_reject_screen(tool_link):
    c,n,local=fixture();local[tool_link]=box([10.,0.,1.])
    assert c.sample(np.zeros(8),.069)==f'empty_pickup_screen:{tool_link}'


def test_tool_robot_collision_and_adjacent_palm_exception():
    c,n,local=fixture();local[PALM_COLLISION_LINK]=box([0.,0.,1.])
    assert 'tool_robot' in c.sample(np.zeros(8),.069)
    c,n,local=fixture();local[PALM_COLLISION_LINK]=box([0.,0.,1.])
    n._world_collision_surfaces=lambda q,**kw:{'arm_left_7_link':box([0.,0.,1.])}
    assert c.sample(np.zeros(8),.069) is None


def test_torso_segment_detects_interior_collision():
    c,n,_=fixture(); n._robot_self_collision=lambda q,**kw:('torso','head') if .16<q[0]<.19 else None
    end=np.zeros(8);end[0]=.35
    assert not c.edge(np.zeros(8),end)
    assert 'robot_self' in c.last_rejection


def test_opening_samples_intermediate_aperture():
    c,n,_=fixture();c.initial_aperture=0.
    seen=[]
    c.sample=lambda q,a: seen.append(float(a)) or ('collision' if .03<a<.04 else None)
    assert not c.opening()
    assert any(.03<a<.04 for a in seen)


def test_candidate_keeps_each_command_leg_and_cartesian_approach():
    c,n,_=fixture();seen=[]
    c.edge=lambda a,b: seen.append((np.array(a),np.array(b))) or True
    t=np.ones(8)*.1;s1=np.ones(8)*.2;s2=np.ones(8)*.3
    assert c.candidate([s1,s2],[t])
    assert len(seen)==3 and seen[0][0][0]==.35
    assert np.array_equal(seen[-1][1],s2)


def attach_feedback(c,n):
    n._lock=threading.Lock(); n.joints={name:0. for name in NAMES};n.joints[MASTER]=.069
    n._joint_stamps_ns={name:1_000_000_000 for name in NAMES}
    n.get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=1_100_000_000))


@pytest.mark.parametrize('name',NAMES)
def test_missing_or_stale_context_is_not_nominal_home(name):
    c,n,_=fixture();attach_feedback(c,n);n._joint_stamps_ns[name]=0
    with pytest.raises(RuntimeError,match='stale'):measured_context(n)


@pytest.mark.parametrize('name,delta',[('head_2_joint',.004),('arm_right_4_joint',.004),('torso_lift_joint',.0021),('arm_left_3_joint',.0081),(MASTER,.00061)])
def test_fresh_dispatch_rejects_changed_context(name,delta):
    c,n,_=fixture();attach_feedback(c,n);n.joints[name]+=delta
    with pytest.raises(RuntimeError):c.require_fresh(np.zeros(8),.069)


def test_frozen_context_not_mutated_by_caller():
    c,n,_=fixture();assert not c.right.flags.writeable and not c.start.flags.writeable


def test_retracted_bound_precedes_expensive_meshes():
    c,n,_=fixture();calls=[]
    n._retracted_transition_is_safe=lambda a,b:False
    c.edge=lambda a,b:calls.append(True) or True
    assert not c.retracted_edge(np.zeros(8),np.ones(8))
    assert not calls


def test_existing_planner_tries_pregrasp_after_new_direct_edge_rejection():
    path=Path(__file__).resolve().parents[1]/'erc_phase1_solution/manipulation_node.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_plan_retracted_transition','_subset_arm_state','_retracted_transition_is_safe')]
    selected=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
        ast.ClassDef(name='Planner',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    scope=dict(np=np,deque=deque,ARM_JOINTS=ARM_JOINTS,PREGRASP=PREGRASP)
    exec(compile(ast.fix_missing_locations(selected),str(path),'exec'),scope)
    planner=scope['Planner']();planner.cartesian_clearance=.45
    planner.chain=SimpleNamespace(link_positions=lambda q:np.zeros((8,3)))
    first=np.zeros(8);last=np.full(8,.3);seen=[]
    def edge(a,b):
        seen.append((np.array(a),np.array(b)))
        return not (np.array_equal(a,first) and np.array_equal(b,last))
    route=planner._plan_retracted_transition(first,last,edge_validator=edge)
    assert len(route)==1
    expected=PREGRASP.copy();expected[0]=last[0]
    np.testing.assert_array_equal(route[0],expected)
    assert len(seen)==3


def test_cached_rejection_restores_exact_witness():
    c,n,_=fixture();n._robot_self_collision=lambda q,**kw:('left','head')
    q=np.full(8,.1);c.sample(q,.069)
    c.last_rejected_q=None;c.last_rejection=None
    assert c.sample(q,.069)
    assert c.last_rejected_q==q.tolist() and c.last_rejection


def test_missing_tool_geometry_fails_closed():
    c,n,local=fixture();local.pop(LEFT_GRIPPER_COLLISION_LINKS[0])
    assert c.sample(np.zeros(8),.069)=='empty_pickup_tool_inventory_invalid'


@pytest.mark.parametrize('aperture',[-.001,.070,float('nan')])
def test_invalid_nominal_command_aperture_rejects(aperture):
    c,n,_=fixture()
    assert c.sample(np.zeros(8),aperture)=='empty_pickup_joint_limit'


def test_cancelled_dispatch_cannot_use_fresh_context():
    c,n,_=fixture();attach_feedback(c,n);n._cancel.set()
    with pytest.raises(RuntimeError,match='cancelled'):c.require_fresh(np.zeros(8),.069)


def test_ordinary_pick_supplies_planner_and_both_validators():
    source=Path(__file__).resolve().parents[1]
    text=(source/'erc_phase1_solution/manipulation_node.py').read_text()
    cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
    calls=[]
    for method in cls.body:
        if not isinstance(method,ast.FunctionDef):continue
        for call in ast.walk(method):
            if isinstance(call,ast.Call) and any(k.arg=='setup_transition_planner' for k in call.keywords):
                calls.append((method.name,{k.arg for k in call.keywords}))
    assert len(calls)==1 and calls[0][0]=='_pick'
    assert {'transition_edge_validator','candidate_validator','setup_transition_planner'} <= calls[0][1]
    assert 'transition_edge_validator=empty_setup_guard.retracted_edge' in text
    assert 'candidate_validator=empty_setup_guard.candidate' in text
    assert 'setup_transition_planner=empty_setup_guard.plan_transition' in text
    assert 'empty_setup_guard.require_fresh(empty_expected, empty_setup_guard.open_aperture)' in text


def test_low_wrist_search_preserves_exact_goal_and_checks_both_segments():
    c,n,_=fixture();first=np.zeros(8);last=np.arange(8,dtype=float)*.1;last[0]=.35
    n._retracted_transition_is_safe=lambda a,b:True
    c.retracted_edge=lambda a,b:False
    c.sample=lambda q,a: 'bad_pregrasp' if np.array_equal(q[1:],PREGRASP[1:]) else None
    seen=[];c.edge=lambda a,b:seen.append((np.array(a),np.array(b))) or True
    route=c.plan_transition(first,last)
    assert len(route)==1 and len(seen)==2
    expected=last.copy();expected[2]=.5;expected[4]=-2.1;expected[6:]=first[6:]
    np.testing.assert_array_equal(route[0],expected)
    np.testing.assert_array_equal(seen[-1][1],last)
    assert c.open_aperture==.069


def test_low_wrist_search_rejects_bad_second_segment_and_is_bounded():
    c,n,_=fixture();first=np.zeros(8);last=np.ones(8);last[0]=.35
    n._retracted_transition_is_safe=lambda a,b:True;c.retracted_edge=lambda a,b:False
    c.sample=lambda q,a:None
    calls=[]
    def edge(a,b):
        calls.append((np.array(a),np.array(b)))
        return not np.array_equal(b,last)
    c.edge=edge
    assert c.plan_transition(first,last) is None
    assert len(calls)==14  # PREGRASP plus six bounded low-wrist templates, both legs.


def test_low_wrist_search_cancel_after_geometry_cannot_return_route():
    c,n,_=fixture();n._retracted_transition_is_safe=lambda a,b:True
    c.retracted_edge=lambda a,b:False;c.sample=lambda q,a:None
    def edge(a,b):n._cancel.set();return True
    c.edge=edge
    assert c.plan_transition(np.zeros(8),np.ones(8)) is None


def test_low_wrist_search_preserves_joint_limits_before_mesh_work():
    c,n,_=fixture();n.chain.lower=np.zeros(8);n.chain.upper=np.ones(8)
    n._retracted_transition_is_safe=lambda a,b:True;c.retracted_edge=lambda a,b:False
    c.sample=lambda q,a:pytest.fail('out-of-limit template reached mesh')
    assert c.plan_transition(np.zeros(8),np.ones(8)) is None


def cartesian_solver():
    path=Path(__file__).resolve().parents[1]/'erc_phase1_solution/manipulation_node.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_solve_cartesian_path','_arm_route_cost')]
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
        ast.ClassDef(name='Solver',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    scope=dict(np=np,HOME=HOME,OFFER=OFFER,PREGRASP=PREGRASP,pose_matrix=lambda p,r:p)
    exec(compile(ast.fix_missing_locations(mod),str(path),'exec'),scope)
    solver=scope['Solver']();solver.position_tolerance=.012;solver.orientation_tolerance=.1;solver.cartesian_joint_step=.4
    solver._current_seed=lambda:np.zeros(8)
    solver.chain=SimpleNamespace(solve=lambda p,seeds,**kw:(np.r_[.35,np.ones(7)*float(p[0])],0.))
    return solver


def test_cartesian_default_still_uses_existing_setup_planner():
    solver=cartesian_solver();seen=[]
    solver._plan_retracted_transition=lambda a,b,**kw:seen.append((a,b,kw)) or []
    result=solver._solve_cartesian_path([[.1,0,0]],[np.eye(3)],.35,first_valid=True)
    assert len(seen)==1 and seen[0][2]=={} and result[-1]==[]


def test_cartesian_callback_is_followed_by_full_candidate_validation():
    solver=cartesian_solver();events=[];waypoint=np.ones(8)*.05;waypoint[0]=.35
    solver._plan_retracted_transition=lambda *a,**kw:pytest.fail('legacy search should not delay scoped callback')
    def setup(a,b):events.append('setup');return [waypoint]
    def validate(solutions,transition):
        events.append('full_candidate');np.testing.assert_array_equal(transition[0],waypoint);return True
    result=solver._solve_cartesian_path([[.1,0,0]],[np.eye(3)],.35,first_valid=True,
        setup_transition_planner=setup,transition_edge_validator=lambda a,b:True,candidate_validator=validate)
    assert events==['setup','full_candidate'];np.testing.assert_array_equal(result[-1][0],waypoint)


@pytest.mark.parametrize('missing',['edge','candidate','skip'])
def test_cartesian_callback_requires_full_guard_and_cannot_skip_setup(missing):
    solver=cartesian_solver();kw=dict(setup_transition_planner=lambda a,b:[],transition_edge_validator=lambda a,b:True,candidate_validator=lambda a,b:True)
    if missing=='edge':kw.pop('transition_edge_validator')
    elif missing=='candidate':kw.pop('candidate_validator')
    else:kw['skip_setup_transition']=True
    with pytest.raises(ValueError,match='requires edge and full candidate'):
        solver._solve_cartesian_path([[.1,0,0]],[np.eye(3)],.35,**kw)


def test_cartesian_full_candidate_rejection_cannot_publish_a_plan():
    solver=cartesian_solver()
    with pytest.raises(RuntimeError,match='No continuous collision-aware'):
        solver._solve_cartesian_path([[.1,0,0]],[np.eye(3)],.35,first_valid=True,
            setup_transition_planner=lambda a,b:[],transition_edge_validator=lambda a,b:True,candidate_validator=lambda a,b:False)
