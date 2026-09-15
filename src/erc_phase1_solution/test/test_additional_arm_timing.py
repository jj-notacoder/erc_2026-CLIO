"""Actual ROS goal arithmetic and actual sender bodies; no physical claim."""
import ast
import copy
from fractions import Fraction
import math
from pathlib import Path

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.duration import Duration
import pytest

from erc_phase1_solution import additional_arm_timing as extra
from erc_phase1_solution import arm_velocity_admission as gate
import test_arm_velocity_admission as fixtures
import test_optional_arm_timing as timing


def load(name, **kwargs):
    return timing.load(name, require_retimed_arm_headroom=extra.require_retimed_arm_headroom, **kwargs)


def make(points):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(fixtures.NAMES)
    for positions, ns in points:
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(ns, 1_000_000_000)
        goal.trajectory.points.append(point)
    return goal


def admit(goal, start=None, limits=None):
    return gate.check_serialized_arm_goal(goal, [0.] * 7 if start is None else start,
                                           [1.] * 7 if limits is None else limits)


@pytest.mark.parametrize('factor', [1., 1.25, 1.5, 2.])
def test_parameter_inclusive_range(factor):
    assert extra.checked_additional_arm_time_scale(factor) == factor


@pytest.mark.parametrize('factor', [True, False, '1.5', None, 0., .999, 2.000001, math.nan, math.inf, -math.inf])
def test_invalid_parameter_rejected(factor):
    with pytest.raises(ValueError):
        extra.checked_additional_arm_time_scale(factor)


def test_default_returns_exact_existing_goal_and_phase_objects():
    goal = make([([.2] * 7, 1_000_000_000)])
    legs = [(object(), .8, 'same')]
    result = extra.retime_admitted_arm_goal(goal, {}, 1., nominal_duration=None, legs=legs)
    assert result[0] is goal and result[1] is legs and result[2] is None


def test_real_ros_owned_goal_preserves_geometry_header_and_all_other_fields():
    goal = make([([-0., .1, .2, .3, .4, .5, .6], 1_000_000_001), ([.01] * 7, 2_000_000_003)])
    goal.trajectory.header.frame_id = 'unchanged-frame'
    before = copy.deepcopy(goal)
    owned, legs, info = extra.retime_admitted_arm_goal(goal, admit(goal), 2., nominal_duration=3.)
    assert owned is not goal and legs is None and goal == before
    for a,b in zip(owned.trajectory.points,goal.trajectory.points):
        assert a is not b and a.positions == b.positions
        assert extra._bits(a.positions) == extra._bits(b.positions)
        assert not a.velocities and not a.accelerations and not a.effort
    stripped = copy.deepcopy(owned)
    for a,b in zip(stripped.trajectory.points,goal.trajectory.points):
        a.time_from_start = copy.deepcopy(b.time_from_start)
    assert stripped == goal
    assert info['minimum_segment_ns'] == 87_500_000 and info['unchanged_nominal_watchdog']


def test_later_joint_delta_not_global_factor_sets_exact_80_percent_ceiling():
    goal = make([([.1]*7, 1_000_000_000), ([1.]*7, 2_000_000_000)])
    owned, _, info = extra.retime_admitted_arm_goal(goal, admit(goal), 2., nominal_duration=3.)
    assert info['segments'][0]['new_segment_ns'] == 500_000_000
    required = extra._ceil((Fraction(1.) - Fraction(.1)) * 1_000_000_000 / Fraction(4,5))
    assert info['segments'][1]['new_segment_ns'] == required
    previous, previous_ns = [0.]*7, 0
    for point in owned.trajectory.points:
        ns = point.time_from_start.sec*1_000_000_000 + point.time_from_start.nanosec
        for first,last in zip(previous,point.positions):
            assert abs(Fraction(last)-Fraction(first))*1_000_000_000 <= Fraction(4,5)*(ns-previous_ns)
        previous,previous_ns = point.positions,ns


def test_one_nanosecond_ceiling_and_floor_are_not_rounded_down():
    goal = make([([0.]*7, 175_000_001)])
    _, _, info = extra.retime_admitted_arm_goal(goal, admit(goal), 2., nominal_duration=1.)
    assert info['new_total_ns'] == 87_500_001
    goal.trajectory.points[0].time_from_start.nanosec = 175_000_000
    _, _, info = extra.retime_admitted_arm_goal(goal, admit(goal), 2., nominal_duration=1.)
    assert info['new_total_ns'] == 87_500_000


def test_nominal_allowance_cannot_be_extended_by_headroom_clamp():
    goal = make([([.9]*7, 1_000_000_000)])
    before = copy.deepcopy(goal)
    with pytest.raises(gate.ArmVelocityAdmissionRejected,match='nominal duration'):
        extra.retime_admitted_arm_goal(goal,admit(goal),2.,nominal_duration=1.)
    assert goal == before


def test_retained_phase_solution_order_and_new_serialized_boundaries():
    q1, q2 = [0., *([.1]*7)], [0., *([.2]*7)]
    legs = [(q1,.8,'lower'),(q2,.8,'tuck')]
    goal = make([([.1]*7,800_000_000),([.2]*7,1_600_000_000)])
    _, retimed, info = extra.retime_admitted_arm_goal(goal,admit(goal),2.,nominal_duration=3.,legs=legs)
    assert retimed[0][0] is q1 and retimed[1][0] is q2
    assert [item[1:] for item in retimed] == [(.4,'lower'),(.4,'tuck')]
    assert legs[0][1] == .8 and info['new_total_ns'] == 800_000_000
    phase = load('_trajectory_leg_at_elapsed_time')
    assert phase(retimed,.39)[1] == 'lower' and phase(retimed,.41)[1] == 'tuck'


@pytest.mark.parametrize('fault',['count','positions','times','record','interpolation'])
def test_phase_or_original_goal_mismatch_is_prepublication_veto(fault):
    goal = make([([.2]*7,800_000_000)])
    legs = [([0., *([.2]*7)],.8,'phase')]
    record = admit(goal)
    if fault == 'count': legs = []
    elif fault == 'positions': legs[0][0][1] = .3
    elif fault == 'times': legs = [(legs[0][0],.7,'phase')]
    elif fault == 'record': record['points'][0]['positions'][0] = .3
    else: goal.trajectory.points[0].velocities = [0.]*7
    with pytest.raises(gate.ArmVelocityAdmissionRejected):
        extra.retime_admitted_arm_goal(goal,record,2.,nominal_duration=3.,legs=legs)


def actual_sender(retained, *, factor=2., gate_override=None, retime_override=None):
    node, events, sent, records, acceptance, clock = fixtures.sender_node(retained=retained)
    node.additional_arm_time_scale = factor
    calls = []
    def checked(node, goal):
        assert node.command.depth == node._lock.depth == 1
        calls.append(copy.deepcopy(goal))
        return (gate_override or gate.require_arm_velocity_locked)(node,goal)
    helper = extra.retime_admitted_arm_goal if retime_override is None else retime_override
    fn = load('_send_retained_arm_trajectory' if retained else '_follow',time=clock,
        require_arm_velocity_locked=checked,retime_admitted_arm_goal=helper)
    goal = fixtures.goal([([.2]*7,800_000_000)])
    legs = [([0., *([.2]*7)],.8,'phase')]
    def run():
        if retained:
            return fn(node,goal,2.4,legs,'place',velocity_admission=True)
        return fn(node,node.arm_client,fixtures.NAMES,[.2]*7,2.4,trajectory_duration=.8)
    return node,events,sent,records,calls,goal,legs,run


@pytest.mark.parametrize('retained',[False,True])
def test_actual_sender_checks_old_and_actual_new_goal_inside_same_locks(retained):
    n,events,sent,records,calls,goal,legs,run = actual_sender(retained)
    assert run() == ((True,False) if retained else True)
    assert len(calls)==2 and len(sent)==len(records)==1
    assert timing.seconds(calls[0].trajectory.points[0].time_from_start)==.8
    assert timing.seconds(calls[1].trajectory.points[0].time_from_start)==.4
    assert vars(sent[0].trajectory)==vars(calls[1].trajectory)
    assert timing.seconds(goal.trajectory.points[0].time_from_start)==.8 and legs[0][1]==.8
    assert records[0]['additional_arm_timing']['new_total_ns']==400_000_000
    assert not n._goal_handles and not n._pending_retained_acceptances


@pytest.mark.parametrize('retained',[False,True])
@pytest.mark.parametrize('when',['first','second'])
def test_either_fresh_gate_failure_prevents_send_and_pending_token(retained,when):
    count=[]
    def check(n,g):
        count.append(1)
        if len(count)==(1 if when=='first' else 2):
            n._joint_stamps_ns[fixtures.NAMES[0]]=fixtures.NOW-150_000_001
        return gate.require_arm_velocity_locked(n,g)
    n,_,sent,records,_,_,_,run = actual_sender(retained,gate_override=check)
    with pytest.raises(gate.ArmVelocityAdmissionRejected): run()
    assert not sent and not n._pending_retained_acceptances and not n._cancel.is_set()
    assert len(records)==1 and not records[0]['admitted']


@pytest.mark.parametrize('retained',[False,True])
def test_actual_second_start_must_still_preserve_80_percent_headroom(retained):
    calls=[]
    def check(n,g):
        calls.append(1)
        if len(calls)==2:
            # Synthetic mutation despite held locks: official <100% still
            # passes, but the optional dispatch's stricter80% must reject.
            n.joints[fixtures.NAMES[0]]=-.5
        return gate.require_arm_velocity_locked(n,g)
    n,_,sent,records,_,_,_,run=actual_sender(retained,gate_override=check)
    with pytest.raises(gate.ArmVelocityAdmissionRejected,match='80-percent'):
        run()
    assert not sent and not n._pending_retained_acceptances
    assert len(records)==1 and not records[0]['admitted']


@pytest.mark.parametrize('retained',[False,True])
def test_factor_one_uses_only_original_gate_and_never_calls_retimer(retained):
    def forbidden(*args,**kwargs): raise AssertionError('default retiming')
    n,_,sent,records,calls,_,_,run=actual_sender(retained,factor=1.,retime_override=forbidden)
    assert run()==((True,False) if retained else True)
    assert len(calls)==1 and timing.seconds(sent[0].trajectory.points[0].time_from_start)==.8
    assert 'additional_arm_timing' not in records[0]


def test_unmarked_default_follow_is_not_newly_opted_in():
    node,sent,_,_,clock=timing.follow_node()
    node.additional_arm_time_scale=2.
    def forbidden(*args,**kwargs): raise AssertionError('unmarked timing changed')
    fn=load('_follow',time=clock,retime_admitted_arm_goal=forbidden)
    assert fn(node,node.arm_client,fixtures.NAMES,[.1]*7,2.8)
    assert timing.seconds(sent[0].trajectory.points[0].time_from_start)==2.8


@pytest.mark.parametrize('retained',[False,True])
def test_cancellation_during_owned_retime_still_prevents_dispatch(retained):
    holder={}
    def cancel(*args,**kwargs):
        result=extra.retime_admitted_arm_goal(*args,**kwargs)
        holder['node']._cancel.set()
        return result
    node,_,sent,records,_,_,_,run=actual_sender(retained,retime_override=cancel)
    holder['node']=node
    with pytest.raises(gate.ArmVelocityAdmissionRejected,match='cancelled before'):
        run()
    assert not sent and not node._pending_retained_acceptances
    assert len(records)==1 and not records[0]['admitted']


def test_retimed_follow_retains_original_wall_watchdog():
    node,sent,_,sleeps,clock=timing.follow_node(late=True)
    node.additional_arm_time_scale=2.
    fn=load('_follow',time=clock,retime_admitted_arm_goal=extra.retime_admitted_arm_goal)
    assert fn(node,node.arm_client,fixtures.NAMES,[.1]*7,2.8,trajectory_duration=2.24)
    assert timing.seconds(sent[0].trajectory.points[0].time_from_start)==1.12
    assert sleeps==[.02] and not node._goal_handles


@pytest.mark.parametrize('retained',[False,True])
def test_original_unsafe_goal_cannot_be_rescued_by_retiming(retained):
    def forbidden(*args,**kwargs): raise AssertionError('unsafe old goal reached retimer')
    node,_,sent,_,calls,_,_,run=actual_sender(retained,retime_override=forbidden)
    node.joints[fixtures.NAMES[0]]=-10.
    with pytest.raises(gate.ArmVelocityAdmissionRejected): run()
    assert len(calls)==1 and not sent and not node._pending_retained_acceptances


def test_pre_send_context_refusal_is_preserved_before_any_retiming():
    node,_,sent,_,_,clock=fixtures.sender_node()
    node.additional_arm_time_scale=2.
    def refused(): raise RuntimeError('scene changed')
    def forbidden(*args,**kwargs): raise AssertionError('context failure reached retimer')
    fn=load('_follow',time=clock,retime_admitted_arm_goal=forbidden)
    with pytest.raises(RuntimeError,match='scene changed'):
        fn(node,node.arm_client,fixtures.NAMES,[.2]*7,2.4,trajectory_duration=.8,pre_send_check=refused)
    assert not sent


def test_unmarked_retained_extension_or_roll_is_unchanged():
    node,_,sent,records,_,clock=fixtures.sender_node(retained=True)
    node.additional_arm_time_scale=2.
    node._place_contact_guard=object()
    def forbidden(*args,**kwargs): raise AssertionError('unmarked retained goal changed')
    fn=load('_send_retained_arm_trajectory',time=clock,retime_admitted_arm_goal=forbidden)
    goal=fixtures.goal()
    assert fn(node,goal,1.,[([0.]*8,1.,'cradle_roll')],'compact_transport')==(True,False)
    assert len(sent)==1 and not records
    assert timing.seconds(sent[0].trajectory.points[0].time_from_start)==1.


def test_retained_fault_phase_uses_actual_new_segment_durations():
    node,_,sent,_,acceptance,clock=fixtures.sender_node(retained=True)
    node.additional_arm_time_scale=2.
    current={'now':fixtures.NOW,'checks':0}
    from types import SimpleNamespace as NS
    node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=current['now']))
    def fault(**kwargs):
        current['checks']+=1
        if current['checks']>=3:
            current['now']=fixtures.NOW+410_000_000
            return 'lost_pressure'
        return None
    node._payload_hazard_reason=fault
    node._trajectory_leg_at_elapsed_time=load('_trajectory_leg_at_elapsed_time')
    acceptance.value.get_result_async=lambda:fixtures.Future(NS(status=4),done=False)
    statuses=[]
    node._publish_status=lambda event,**data:statuses.append((event,data))
    q1,q2=[0., *([.1]*7)],[0., *([.2]*7)]
    legs=[(q1,.8,'lower'),(q2,.8,'tuck')]
    goal=fixtures.goal([([.1]*7,800_000_000),([.2]*7,1_600_000_000)])
    fn=load('_send_retained_arm_trajectory',time=clock,retime_admitted_arm_goal=extra.retime_admitted_arm_goal)
    assert fn(node,goal,4.8,legs,'place',velocity_admission=True)==(False,True)
    lost=[data for event,data in statuses if event=='grasp_lost']
    assert len(sent)==1 and lost[0]['phase']=='tuck' and lost[0]['leg']==1
    assert legs[0][1]==.8 and not node._goal_handles


def test_actual_constructor_parameter_wiring_rejects_boolean_and_keeps_default():
    init=timing.method_ast('__init__')
    assignment=next(n for n in ast.walk(init) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Attribute) and t.attr=='additional_arm_time_scale' for t in n.targets))
    from types import SimpleNamespace as NS
    for value in (1.,1.5,2.,True):
        node=NS(get_parameter=lambda name:NS(value=value))
        scope={'self':node,'checked_additional_arm_time_scale':extra.checked_additional_arm_time_scale}
        block=compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'<actual init assignment>','exec')
        if value is True:
            with pytest.raises(ValueError): exec(block,scope)
        else:
            exec(block,scope); assert node.additional_arm_time_scale==value
    declaration=timing.method_ast('_declare_parameters')
    assert any(isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='additional_arm_time_scale'
        and isinstance(v,ast.Constant) and v.value==1. for k,v in zip(n.keys,n.values)) for n in ast.walk(declaration))
