"""Actual compact method and ROS goal serialization with controlled sensors.

Geometry/probes/action results are deterministic fixtures; no physical safety or
acceleration proof is claimed. Whole-parent inverses preserve the original gates.
"""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
from control_msgs.action import FollowJointTrajectory
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectoryPoint

from erc_phase1_solution.arm_trajectory_timing import checked_arm_speed_scale, scaled_arm_seconds
from erc_phase1_solution.arm_velocity_admission import ArmVelocityAdmissionRejected
from erc_phase1_solution.compact_timing import checked_supported_compact_minimum_seconds
from erc_phase1_solution.motion_profiles import ARM_JOINTS
from candidate_composition_support import restore_supported_compact_floor_bytes

SOURCE=Path(__file__).resolve().parents[1]
NODE=SOURCE/'erc_phase1_solution/manipulation_node.py'


def method(name, parent=False):
    raw=NODE.read_bytes()
    if parent:raw=restore_supported_compact_floor_bytes(raw)
    tree=ast.parse(raw)
    owner=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
    return copy.deepcopy(next(x for x in owner.body if isinstance(x,ast.FunctionDef) and x.name==name))


def load(name,parent=False):
    fn=method(name,parent);fn.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    scope=dict(np=np,math=math,Duration=Duration,JointTrajectoryPoint=JointTrajectoryPoint,
        FollowJointTrajectory=FollowJointTrajectory,ARM_JOINTS=ARM_JOINTS,
        checked_arm_speed_scale=checked_arm_speed_scale,scaled_arm_seconds=scaled_arm_seconds,
        checked_supported_compact_minimum_seconds=checked_supported_compact_minimum_seconds)
    exec(compile(ast.fix_missing_locations(module),'<actual compact method>','exec'),scope)
    return scope[name]


@pytest.mark.parametrize('value',[.175,math.nextafter(.175,math.inf),.2,.275,math.nextafter(.35,-math.inf),.35])
def test_valid_inclusive_floor(value):
    assert checked_supported_compact_minimum_seconds(value)==value


@pytest.mark.parametrize('value',[0.,-.1,.174999,math.nextafter(.175,-math.inf),.350001,math.nextafter(.35,math.inf),float('nan'),float('inf'),-float('inf'),True,False])
def test_invalid_floor_rejected(value):
    with pytest.raises(ValueError):checked_supported_compact_minimum_seconds(value)


@pytest.mark.parametrize('value',[.175,.35,float('nan'),.1,.4])
def test_actual_constructor_assignment_uses_bounded_parameter(value):
    fn=method('__init__')
    assignments=[x for x in fn.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='supported_compact_minimum_segment_seconds' for t in x.targets)]
    assert len(assignments)==1
    names=[]
    n=NS(get_parameter=lambda name:(names.append(name) or NS(value=value)))
    scope=dict(self=n,checked_supported_compact_minimum_seconds=checked_supported_compact_minimum_seconds)
    code=compile(ast.fix_missing_locations(ast.Module(body=assignments,type_ignores=[])),'<actual constructor parameter>','exec')
    if value in (.175,.35):
        exec(code,scope);assert n.supported_compact_minimum_segment_seconds==value
    else:
        with pytest.raises(ValueError):exec(code,scope)
        assert not hasattr(n,'supported_compact_minimum_segment_seconds')
    assert names==['supported_compact_minimum_segment_seconds']


def scaling_block(groups,scale,floor=None):
    fn=method('_execute_cached_post_retreat_compaction')
    start=next(i for i,x in enumerate(fn.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='supported_watchdog_duration' for t in x.targets))
    n=NS(placement_transport_speed_scale=scale)
    if floor is not None:n.supported_compact_minimum_segment_seconds=floor
    scope=dict(self=n,timed_groups=groups,checked_arm_speed_scale=checked_arm_speed_scale,scaled_arm_seconds=scaled_arm_seconds)
    exec(compile(ast.fix_missing_locations(ast.Module(body=fn.body[start:start+3],type_ignores=[])),'<actual supported timing block>','exec'),scope)
    return scope


@pytest.mark.parametrize('scale',[1.,1.25,2.,3.])
@pytest.mark.parametrize('floor',[.175,.35])
def test_actual_block_no_stacking_no_other_group_change(scale,floor):
    q=np.arange(8,dtype=float)
    groups=[[(q,.35,'extension')],[(q,.75,'roll')],[(q,.35,'lower'),(q,.8,'retraction'),(q,2.8,'compact')]]
    first,second=groups[:2]
    scope=scaling_block(groups,scale,floor)
    assert groups[0] is first and groups[1] is second
    expected=[.35,.8,2.8] if scale==1 else [max(floor,x/scale) for x in (.35,.8,2.8)]
    assert [x[1] for x in groups[2]]==expected
    assert all(x[0] is q for group in groups for x in group)
    assert [x[2] for x in groups[2]]==['lower','retraction','compact']
    assert scope['supported_watchdog_duration']==(None if scale==1 else sum((.35,.8,2.8)))


@pytest.mark.parametrize('floor',[.1,.4,float('nan'),float('inf')])
def test_invalid_runtime_floor_stops_before_replacing_any_group(floor):
    q=np.zeros(8);groups=[[(q,.35,'e')],[(q,.75,'r')],[(q,.35,'s')]]
    originals=list(groups)
    with pytest.raises(ValueError):scaling_block(groups,3.,floor)
    assert all(x is y for x,y in zip(groups,originals))


def compact_fixture(floor=.35,scale=3.,fault=None):
    q=np.array([.35,0.,0.,0.,0.,0.,0.,0.])
    phases=['post_retreat_clearance_extension','post_retreat_cradle_roll']+['supported_cradle_lowering']*8+['supported_cradle_retraction']*8+['compact_transport']*8
    raw=[]
    for i,phase in enumerate(phases):
        q=q.copy();q[1]+=.4 if i==13 else .04
        raw.append((q,phase))
    initial=np.array([.35,0.,0.,0.,0.,0.,0.,0.]);corners=np.zeros((8,3))
    n=NS(_cached_post_retreat_plan=dict(requires_gravity_support=False,start=initial.copy(),shelf_front_x=.8,attached_corners=corners.copy(),legs=raw,staging_terminal=raw[17][0].copy(),terminal=raw[-1][0].copy(),compact_radius=.2),
        _held_book_corners=corners,_gravity_supported_payload=False,_retention_probe_active=False,_payload_robot_watchdog_enabled=False,
        carried_navigation_radius_limit=.5,carried_cradle_transfer=.75,timeout=20.,_lock=threading.RLock(),
        placement_transport_speed_scale=scale,supported_compact_minimum_segment_seconds=floor)
    sends=[];probes=[];endpoints=[];statuses=[];geometry=[]
    n.arm_client=NS(wait_for_server=lambda **kw:fault!='server')
    n._transport_leg_duration=load('_transport_leg_duration')
    n._make_retained_arm_trajectory_goal=lambda legs:load('_make_retained_arm_trajectory_goal')(n,legs)
    def safe(a,b,*args):
        geometry.append((a.copy(),b.copy()))
        return not (fault=='initial_geometry' and len(geometry)==1) and not (fault=='final_geometry' and len(geometry)>2)
    n._carried_post_retreat_transition_is_safe=safe
    measured=iter([initial.copy(),raw[0][0].copy(),raw[1][0].copy()])
    def read():
        answer=next(measured)
        if fault=='post_roll_drift' and np.array_equal(answer,raw[1][0]):answer=answer.copy();answer[1]+=.006
        return answer
    n._measured_left_solution=read
    def probe(command,phase,**kw):
        probes.append(phase)
        return phase!=fault
    n._fresh_retention_probe=probe
    n._retention_after_leg=lambda *args:fault!='retention_after_roll'
    def send(goal,duration,legs,command,**kw):
        sends.append(dict(goal=goal,duration=duration,legs=legs,command=command,kwargs=kw,gravity=n._gravity_supported_payload))
        if len(sends)==3 and fault=='velocity':raise ArmVelocityAdmissionRejected('actual-final-gate-veto')
        return (not (fault=='send_'+str(len(sends))),False)
    n._send_retained_arm_trajectory=send
    def endpoint(q,**kw):
        endpoints.append(kw['phase'])
        return None if fault=='endpoint_'+str(len(endpoints)) else q.copy()
    n._wait_for_retained_endpoint=endpoint
    n._gravity_supported_transition_is_safe=lambda *a:True
    n._carried_navigation_radius=lambda *a:.2
    n._publish_status=lambda *a,**kw:statuses.append((a,kw))
    return n,initial,raw,sends,probes,endpoints,statuses,geometry


def stamps(goal):
    return [p.time_from_start.sec*1_000_000_000+p.time_from_start.nanosec for p in goal.trajectory.points]


def test_actual_full_flow_changes_only_postroll_times_preserves_ros_positions_and_watchdog():
    records=[]
    for floor in (.35,.175):
        n,start,raw,sends,probes,endpoints,statuses,geometry=compact_fixture(floor)
        assert load('_execute_cached_post_retreat_compaction')(n,start,.8) is True
        assert len(sends)==3 and len(sends[2]['legs'])==24
        assert [x['kwargs'] for x in sends]==[{}, {}, {'velocity_admission':True}]
        assert [x['gravity'] for x in sends]==[False,True,True]
        assert probes==['post_retreat','post_retreat_clearance_extension','post_retreat_cradle_roll','compact_transport_final']
        assert len(endpoints)==3 and len(geometry)==3
        assert n._cached_post_retreat_plan is None and n._supported_post_retreat_staging_required is False
        assert not n._retention_probe_active and not n._payload_robot_watchdog_enabled
        assert statuses[-1][0]==('transport_compact',)
        records.append(sends)
    baseline,fast=records
    assert [stamps(x['goal']) for x in baseline[:2]]==[stamps(x['goal']) for x in fast[:2]]
    for a,b in zip(baseline,fast):
        assert a['duration']==b['duration']  # Original unscaled watchdog retained.
        assert a['goal'].trajectory.joint_names==b['goal'].trajectory.joint_names==list(ARM_JOINTS)
        assert [x.positions for x in a['goal'].trajectory.points]==[x.positions for x in b['goal'].trajectory.points]
        assert [x[2] for x in a['legs']]==[x[2] for x in b['legs']]
    assert stamps(fast[2]['goal'])[-1]<stamps(baseline[2]['goal'])[-1]
    times=stamps(fast[2]['goal'])
    assert all(b>a for a,b in zip([0]+times,times))
    assert all(not p.velocities and not p.accelerations and not p.effort for x in fast for p in x['goal'].trajectory.points)


@pytest.mark.parametrize('scale',[1.,1.25,3.])
def test_default_full_flow_exact_against_parent_ros_goals(scale):
    captured=[]
    for parent in (True,False):
        n,start,raw,sends,probes,endpoints,statuses,geometry=compact_fixture(.35,scale)
        assert load('_execute_cached_post_retreat_compaction',parent)(n,start,.8)
        captured.append((sends,probes,endpoints,statuses,geometry))
    old,new=captured
    assert old[1:4]==new[1:4]
    for a,b in zip(old[0],new[0]):
        assert a['goal']==b['goal'] and a['duration']==b['duration'] and a['kwargs']==b['kwargs']
    for a,b in zip(old[4],new[4]):
        assert all(np.array_equal(x,y) for x,y in zip(a,b))


@pytest.mark.parametrize('fault,sends',[('server',0),('initial_geometry',0),('post_retreat',0),('send_1',1),('endpoint_1',1),('post_retreat_clearance_extension',1),('send_2',2),('endpoint_2',2),('retention_after_roll',2),('post_retreat_cradle_roll',2),('post_roll_drift',2),('send_3',3),('endpoint_3',3),('final_geometry',3),('compact_transport_final',3)])
def test_actual_flow_retains_failure_boundaries(fault,sends):
    n,start,raw,seen,_,_,statuses,_=compact_fixture(.175,fault=fault)
    with pytest.raises(RuntimeError):load('_execute_cached_post_retreat_compaction')(n,start,.8)
    assert len(seen)==sends
    assert n._cached_post_retreat_plan is not None
    assert not statuses
    assert not n._retention_probe_active and not n._payload_robot_watchdog_enabled


def test_actual_flow_velocity_rejection_propagates_and_does_not_mark_complete():
    n,start,raw,seen,_,_,statuses,_=compact_fixture(.175,fault='velocity')
    with pytest.raises(ArmVelocityAdmissionRejected,match='actual-final-gate-veto'):
        load('_execute_cached_post_retreat_compaction')(n,start,.8)
    assert len(seen)==3 and seen[-1]['kwargs']=={'velocity_admission':True}
    assert n._cached_post_retreat_plan is not None and not statuses


def test_default_declaration_and_complete_parent_inverse():
    declaration=method('_declare_parameters')
    dictionaries=[x for x in ast.walk(declaration) if isinstance(x,ast.Dict)]
    defaults={k.value:v.value for x in dictionaries for k,v in zip(x.keys,x.values) if isinstance(k,ast.Constant) and isinstance(v,ast.Constant)}
    assert defaults['supported_compact_minimum_segment_seconds']==.35
    original=restore_supported_compact_floor_bytes(NODE.read_bytes())
    assert hashlib.sha256(original).hexdigest()=='52e1d83329bf69925352a219670ac1a42bd33abddce3f7e1b81797dc035eb581'
    for name in ('_transport_leg_duration','_make_retained_arm_trajectory_goal','_send_retained_arm_trajectory','_execute_retained_arm_legs','_place','_return_from_bin','_execute_unloaded_home'):
        from candidate_composition_support import restore_current_extensions
        owner=next(n for n in ast.parse(restore_current_extensions(NODE.read_bytes())).body if isinstance(n,ast.ClassDef) and n.name=="ManipulationNode")
        historical=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(historical,include_attributes=False)==ast.dump(method(name,True),include_attributes=False)


@pytest.mark.parametrize('mutation',[lambda b:b+b'\n# undeclared\n',lambda b:b.replace(b'minimum=supported_minimum',b'minimum=.1'),lambda b:b.replace(b"'supported_compact_minimum_segment_seconds': .35",b"'supported_compact_minimum_segment_seconds': .175")])
def test_inverse_rejects_undeclared_changes(mutation):
    with pytest.raises(AssertionError):restore_supported_compact_floor_bytes(mutation(NODE.read_bytes()))
