"""Actual ROS goals and production sender/empty-context gates; no physics claim."""
import ast
import copy
from fractions import Fraction
from pathlib import Path

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.duration import Duration
import numpy as np
import pytest

from erc_phase1_solution import empty_pickup_setup_timing as setup
from erc_phase1_solution import additional_arm_timing as extra
from erc_phase1_solution.arm_velocity_admission import ArmVelocityAdmissionRejected
from erc_phase1_solution.empty_pickup_collision import EmptyPickupCollision, NAMES, HEAD, MASTER
from erc_phase1_solution.motion_profiles import ARM_JOINTS, RIGHT_ARM_JOINTS
import test_optional_arm_timing as fixtures

ROOT=Path(__file__).resolve().parents[1]


def harness(*, enabled=True, global_factor=1., late=False, retimer=None):
    n,sent,statuses,sleeps,clock=fixtures.follow_node(late=late)
    n.empty_pickup_setup_retiming_enabled=enabled
    n.additional_arm_time_scale=global_factor
    n._held_book_corners=None
    n.gripper_open=.069
    n.adaptive_endpoint_tolerance=.001
    n.joints.update(dict.fromkeys(NAMES,0.))
    n.joints['torso_lift_joint']=.35
    n.joints[MASTER]=.069
    n._joint_stamps_ns=dict.fromkeys(NAMES,1_000_000_000)
    expected=np.array([.35,*([0.]*7)])
    guard=EmptyPickupCollision(n,expected,[0.]*7,[0.]*2,.069,
        geometry=object(),screen=object())
    follow=fixtures.load('_follow',time=clock,Duration=Duration,
        JointTrajectoryPoint=JointTrajectoryPoint,FollowJointTrajectory=FollowJointTrajectory,
        retime_admitted_arm_goal=extra.retime_admitted_arm_goal if retimer is None else retimer,
        require_retimed_arm_headroom=extra.require_retimed_arm_headroom)
    n._follow=lambda *a,**k:follow(n,*a,**k)
    original=fixtures.load('_move_arm_solution')
    n._move_arm_solution=lambda *a,**k:original(n,*a,**k)
    return n,guard,expected,sent,statuses,sleeps


@pytest.mark.parametrize('value',[False,True])
def test_only_bool_parameter(value):
    assert setup.checked_empty_pickup_setup_retiming_enabled(value) is value


@pytest.mark.parametrize('value',[0,1,2,'true',None,[],float('nan')])
def test_non_bool_parameter_rejected(value):
    with pytest.raises(ValueError):setup.checked_empty_pickup_setup_retiming_enabled(value)


@pytest.mark.parametrize('factor',[1.,2.])
def test_per_call_factor_two_has_real_owned_goal_and_no_global_stacking(factor):
    retained=[]
    def observe(goal,admission,*a,**k):
        before=copy.deepcopy(goal)
        owned,legs,info=extra.retime_admitted_arm_goal(goal,admission,*a,**k)
        retained.append((goal,before,owned,admission,info))
        return owned,legs,info
    n,g,q,sent,statuses,_=harness(global_factor=factor,retimer=observe)
    target=q.copy();target[1]=1.5
    before=target.copy()
    assert setup.move_empty_pickup_setup(n,g,q,target,2.2,phase='empty_setup_transition')
    assert np.array_equal(target,before) and len(retained)==len(sent)==1
    goal,snapshot,owned,admission,info=retained[0]
    assert goal==snapshot and goal is not owned and sent[0] is owned
    assert goal.trajectory.points[0] is not owned.trajectory.points[0]
    assert extra._bits(goal.trajectory.points[0].positions)==extra._bits(owned.trajectory.points[0].positions)
    assert fixtures.seconds(goal.trajectory.points[0].time_from_start)==2.2
    assert fixtures.seconds(owned.trajectory.points[0].time_from_start)==1.1
    assert info['factor']==2. and info['unchanged_nominal_watchdog']
    assert statuses[-1][1]['command']=='pick' and statuses[-1][1]['admitted']
    assert not n._goal_handles


def test_real_velocity_floor_uses_exact_80_percent_ceiling():
    n,g,q,sent,statuses,_=harness()
    target=q.copy();target[2]=2.33
    assert setup.move_empty_pickup_setup(n,g,q,target,2.2,phase='empty_setup_transition')
    point=sent[0].trajectory.points[0]
    ns=point.time_from_start.sec*1_000_000_000+point.time_from_start.nanosec
    required=Fraction.from_float(2.33)*1_000_000_000/(Fraction(4,5)*Fraction.from_float(1.95))
    assert ns==-(-required.numerator//required.denominator)
    assert statuses[-1][1]['maximum_commanded_velocity_limit_ratio']<=.8


@pytest.mark.parametrize('route',['disabled','alternate','custom'])
def test_default_and_nonordinary_routes_keep_original_goal_and_no_admission(route):
    n,g,q,sent,statuses,_=harness(enabled=route!='disabled',global_factor=2.)
    if route=='alternate':g=None
    elif route=='custom':g=object()
    target=q.copy();target[1]=.1
    assert setup.move_empty_pickup_setup(n,g,q,target,2.2,phase='empty_setup_transition')
    assert fixtures.seconds(sent[0].trajectory.points[0].time_from_start)==2.2
    assert not statuses and not n._goal_handles


@pytest.mark.parametrize('fault',['payload','cancel','stale_right','stale_arm','head_drift','right_drift','torso_drift','aperture','wrong_owner','torso_target','phase','duration'])
def test_unadmitted_empty_context_never_sends(fault):
    n,g,q,sent,_,_=harness()
    target=q.copy();target[1]=.1
    phase,duration='empty_setup_transition',2.2
    if fault=='payload':n._held_book_corners=object()
    elif fault=='cancel':n._cancel.set()
    elif fault=='stale_right':n._joint_stamps_ns[RIGHT_ARM_JOINTS[0]]-=350_000_001
    elif fault=='stale_arm':n._joint_stamps_ns[ARM_JOINTS[0]]-=200_000_000
    elif fault=='head_drift':n.joints[HEAD[0]]=.004
    elif fault=='right_drift':n.joints[RIGHT_ARM_JOINTS[0]]=.004
    elif fault=='torso_drift':n.joints['torso_lift_joint']+=.0021
    elif fault=='aperture':n.joints[MASTER]=.06
    elif fault=='wrong_owner':g.node=object()
    elif fault=='torso_target':target[0]+=.01
    elif fault=='phase':phase='empty_cartesian_approach'
    elif fault=='duration':duration=.65
    with pytest.raises((RuntimeError,ValueError)):
        setup.move_empty_pickup_setup(n,g,q,target,duration,phase=phase)
    assert not sent and not n._goal_handles


def test_context_is_rechecked_after_action_server_wait():
    n,g,q,sent,_,_=harness()
    n.arm_client.wait_for_server=lambda **k:(n.joints.__setitem__(HEAD[0],.004) or True)
    with pytest.raises(RuntimeError,match='context changed'):
        setup.move_empty_pickup_setup(n,g,q,q,2.2,phase='empty_setup_transition')
    assert not sent


def test_second_locked_admission_rejects_goal_changed_by_retirement_boundary():
    def break_goal(goal,admission,*a,**k):
        result=extra.retime_admitted_arm_goal(goal,admission,*a,**k)
        result[0].trajectory.points[0].positions[0]=100.
        return result
    n,g,q,sent,statuses,_=harness(retimer=break_goal)
    with pytest.raises(ArmVelocityAdmissionRejected,match='exceeds official limit'):
        setup.move_empty_pickup_setup(n,g,q,q,2.2,phase='empty_setup_transition')
    assert not sent and statuses[-1][1]['admitted'] is False


def test_original_watchdog_and_registered_action_cleanup_are_used():
    n,g,q,sent,_,sleeps=harness(late=True)
    assert setup.move_empty_pickup_setup(n,g,q,q,2.8,phase='empty_setup_clearance')
    assert fixtures.seconds(sent[0].trajectory.points[0].time_from_start)==1.4
    # The supplied future becomes done at wall11, inside nominal2.8 watchdog12.2
    # but outside shortened1.4 watchdog6.6. Original sender succeeds and unregisters.
    assert sleeps==[.02] and not n._goal_handles


def test_unmarked_hold_or_recovery_cannot_inherit_setup_retiming():
    n,g,q,sent,statuses,_=harness(global_factor=1.)
    assert n._move_arm_solution(q,2.8)
    assert fixtures.seconds(sent[0].trajectory.points[0].time_from_start)==2.8
    assert not statuses
    # There is no transient per-node marker to restore on success or exception.
    assert not hasattr(n,'empty_pickup_setup')


def test_scope_preserves_final_approach_and_all_existing_endpoint_calls():
    text=(ROOT/'erc_phase1_solution/manipulation_node.py').read_text()
    owner=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
    pick=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_pick')
    calls=[n for n in ast.walk(pick) if isinstance(n,ast.Call)]
    opted=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='move_empty_pickup_setup']
    assert len(opted)==2
    assert {k.value.value for n in opted for k in n.keywords if k.arg=='phase'}=={'empty_setup_transition','empty_setup_clearance'}
    final=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='_move_arm_solution'
           and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value==.65]
    assert len(final)==1 and not final[0].keywords
    waits=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='wait_for_endpoint']
    assert {k.value.value for n in waits for k in n.keywords if k.arg=='phase'}=={
        'empty_setup_torso','empty_setup_transition','empty_setup_clearance','empty_cartesian_approach'}
    assert "'empty_pickup_setup_retiming_enabled': False" in text
    init=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    initialization=[n for n in ast.walk(init) if isinstance(n,ast.If)
                    and 'self._faster_arm_velocity_limits' in ast.unparse(n)]
    assert any('self.empty_pickup_setup_retiming_enabled' in ast.unparse(n.test) for n in initialization)


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_optional_only_constructor_branch_loads_official_limits(enabled,tmp_path,monkeypatch):
    from types import SimpleNamespace as NS
    from erc_phase1_solution import empty_pickup_collision as collision
    from erc_phase1_solution.motion_profiles import IK_JOINTS
    method=fixtures.method_ast('__init__')
    branch=next(n for n in ast.walk(method) if isinstance(n,ast.If)
                and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Attribute)
                    and t.attr=='_faster_arm_velocity_limits' for t in x.targets) for x in n.body))
    urdf=tmp_path/'velocity_fixture.urdf'
    limits=[.035,1.95,1.95,3.95,3.95,3.95,3.95,3.95]
    urdf.write_text('<robot name="fixture">'+''.join(
        f'<joint name="{name}" type="revolute"><limit velocity="{limit}"/></joint>'
        for name,limit in zip(IK_JOINTS,limits))+'</robot>')
    actual=collision.joint_velocity_limits;calls=[]
    def read(path):
        calls.append(path)
        return actual(path)
    monkeypatch.setattr(collision,'joint_velocity_limits',read)
    node=NS(placement_transport_speed_scale=1.,withdrawal_speed_scale=1.,
            empty_pickup_setup_retiming_enabled=enabled)
    scope=dict(self=node,urdf=urdf,__package__='erc_phase1_solution')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),
                 '<actual optional-only limit constructor>','exec'),scope)
    if enabled:
        assert calls==[urdf]
        assert node._faster_arm_velocity_limits==tuple(limits[1:])
        assert node._faster_arm_velocity_urdf==str(urdf)
    else:
        assert calls==[] and not hasattr(node,'_faster_arm_velocity_limits')


@pytest.mark.parametrize('value',[False,True,0,1,'true',None])
def test_actual_flag_constructor_executes_strict_checker(value):
    from types import SimpleNamespace as NS
    method=fixtures.method_ast('__init__')
    assignment=next(n for n in ast.walk(method) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Attribute) and t.attr=='empty_pickup_setup_retiming_enabled' for t in n.targets))
    calls=[]
    node=NS(get_parameter=lambda name:(calls.append(name) or NS(value=value)))
    scope=dict(self=node,checked_empty_pickup_setup_retiming_enabled=setup.checked_empty_pickup_setup_retiming_enabled)
    code=compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),
                 '<actual empty setup parameter constructor>','exec')
    if type(value) is bool:
        exec(code,scope);assert node.empty_pickup_setup_retiming_enabled is value
    else:
        with pytest.raises(ValueError):exec(code,scope)
    assert calls==['empty_pickup_setup_retiming_enabled']
