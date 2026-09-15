"""Actual yaw-law/stop-order checks; synthetic commands are not tracking or timing proof."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

pytest.importorskip('rclpy')
from rclpy.time import Time
from erc_phase1_solution import navigation_node as nav
import test_normal_linear_convergence as normal

ROOT=Path(__file__).resolve().parents[1]
PARENT_NAV_SHA256='364397dd1c780f162f62450515ca1925201a0d7a8eaea3c3363388ff84302cc0'


def previous_source():
    raw=(ROOT/'erc_phase1_solution/navigation_node.py').read_bytes()
    record=json.loads((ROOT/'test/fixtures/normal_linear_convergence_inverse.json').read_bytes())
    for edit in reversed(record['replacements'][-4:]):
        assert raw.count(edit['new'].encode())==1
        raw=raw.replace(edit['new'].encode(),edit['old'].encode(),1)
    assert hashlib.sha256(raw).hexdigest()==PARENT_NAV_SHA256
    return raw


def previous_method(name):
    owner=next(n for n in ast.parse(previous_source()).body if isinstance(n,ast.ClassDef) and n.name=='NavigationNode')
    method=copy.deepcopy(next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    scope=dict(nav.__dict__)
    exec(compile(ast.fix_missing_locations(module),'<actual prior yaw controller>','exec'),scope)
    return scope[name]


def node(gain=1.8,enabled=True,profile='normal'):
    n=normal.control_node(enabled=enabled,profile=profile)
    n.normal_yaw_gain=nav._checked_normal_yaw_gain(gain)
    return n


def vector(command):
    return command.linear.x,command.linear.y,command.angular.z


@pytest.mark.parametrize('gain',[1.35,1.5,1.8,2,math.nextafter(1.35,math.inf),math.nextafter(2.,0.)])
def test_yaw_valid_range_uses_actual_constructor_assignment(gain):
    owner=next(n for n in ast.parse((ROOT/'erc_phase1_solution/navigation_node.py').read_bytes()).body if isinstance(n,ast.ClassDef) and n.name=='NavigationNode')
    init=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    selected=[n for n in init.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='normal_yaw_gain' for t in n.targets)]
    assert len(selected)==1
    names=[]
    receiver=NS(get_parameter=lambda name:(names.append(name) or NS(value=gain)))
    scope=dict(nav.__dict__,self=receiver)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),'<actual yaw parameter assignment>','exec'),scope)
    assert names==['normal_yaw_gain'] and type(receiver.normal_yaw_gain) is float
    assert receiver.normal_yaw_gain==float(gain)


@pytest.mark.parametrize('gain',[True,False,None,'1.8',math.nan,math.inf,-math.inf,0.,-1.,math.nextafter(1.35,0.),math.nextafter(2.,math.inf),2.001,10**1000])
def test_yaw_rejects_bool_nonfinite_and_values_outside_explicit_range(gain):
    with pytest.raises(ValueError,match='normal_yaw_gain'):
        nav._checked_normal_yaw_gain(gain)


def test_yaw_default_and_existing_controller_limits_are_unchanged():
    values={}
    nav.NavigationNode._declare_parameters(NS(declare_parameter=lambda name,value:values.update({name:value})))
    assert values['normal_yaw_gain']==1.35 and values['normal_linear_convergence_enabled'] is False
    assert {k:values[k] for k in ('control_rate_hz','max_angular_speed','angular_acceleration_limit','yaw_tolerance','settle_time_seconds','emergency_stop_distance')}==dict(control_rate_hz=20.,max_angular_speed=.55,angular_acceleration_limit=.8,yaw_tolerance=.045,settle_time_seconds=.45,emergency_stop_distance=.28)


def test_yaw_changes_only_parameter_assignment_declaration_and_normal_control_multiplier():
    def methods(raw):
        owner=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='NavigationNode')
        return {n.name:ast.dump(n,include_attributes=False) for n in owner.body if isinstance(n,ast.FunctionDef)}
    before=methods(previous_source());after=methods((ROOT/'erc_phase1_solution/navigation_node.py').read_bytes())
    assert before.keys()==after.keys()
    assert {n for n in before if before[n]!=after[n]}=={'__init__','_declare_parameters','_control'}
    for name in ('_desired_translation','_limited_command','_control_empty_arm','_on_command','_sensors_fresh'):
        assert before[name]==after[name]


@pytest.mark.parametrize('yaw',[-.3,0.,.3])
@pytest.mark.parametrize('setting',['missing',1.35])
def test_yaw_default_complete_command_trace_matches_exact_parent(yaw,setting):
    traces=[]
    for function in (nav.NavigationNode._control,previous_method('_control')):
        n=normal.control_node();n.goal=(.2,.1,yaw)
        if setting!='missing':n.normal_yaw_gain=setting
        for _ in range(10):function(n);normal.advance(n)
        traces.append(([vector(c) for c in n.commands],n.goal,n.statuses,n.settle_started))
    assert traces[0]==traces[1]


@pytest.mark.parametrize('profile,enabled',[('normal',False),('carried_retreat',True),('custom',True)])
@pytest.mark.parametrize('yaw',[-.2,.2])
def test_yaw_selected_parameter_is_inert_outside_enabled_normal_profile(profile,enabled,yaw):
    traces=[]
    for function in (nav.NavigationNode._control,previous_method('_control')):
        n=node(profile=profile,enabled=enabled);n.goal=(.12,0.,yaw)
        for _ in range(8):function(n);normal.advance(n)
        traces.append(([vector(c) for c in n.commands],n.goal,n.statuses))
    assert traces[0]==traces[1]


@pytest.mark.parametrize('profile',['empty_arm_staging','empty_arm_advance'])
def test_yaw_does_not_enter_normal_control_for_empty_arm_profiles(profile):
    n=node(profile=profile);calls=[];n._control_empty_arm=lambda now:calls.append(now.nanoseconds)
    n._control()
    assert calls==[n.now_ns] and n.commands==[]


@pytest.mark.parametrize('error,expected',[(.05,.09),(.1,.18),(.2,.36),(.4,.55),(-.1,-.18),(-1.,-.55)])
def test_yaw_actual_normal_commands_keep_direction_and_existing_angular_cap(error,expected):
    n=node();n.goal=(.1,.05,error)
    n.last_command.linear.x=.085;n.last_command.linear.y=.0425;n.last_command.angular.z=expected
    n._control()
    assert vector(n.commands[-1])==pytest.approx((.085,.0425,expected),abs=1e-14)
    assert abs(n.commands[-1].angular.z)<=.55


@pytest.mark.parametrize('sign',[1.,-1.])
@pytest.mark.parametrize('offset',[-1e-8,0.,1e-8])
def test_yaw_saturation_boundary_and_angular_limiter_remain_original(sign,offset):
    n=node();n.goal=(0.,0.,sign*(.55/1.8+offset));n.last_command.angular.z=sign*.55
    n._control();value=n.commands[-1].angular.z
    assert value*sign>0 and abs(value)<=.55
    if offset<0:assert .5499999<abs(value)<.55
    else:assert abs(value)==pytest.approx(.55,abs=1e-14)
    n=node();n.goal=(0.,0.,sign*1.);n._control()
    assert n.commands[-1].angular.z==pytest.approx(sign*.8/20)


@pytest.mark.parametrize('pose_yaw,goal_yaw,sign',[(math.pi-.05,-math.pi+.05,1.),(-math.pi+.05,math.pi-.05,-1.)])
def test_yaw_wrap_uses_shortest_angle_without_larger_command_step(pose_yaw,goal_yaw,sign):
    n=node();n.pose=(0.,0.,pose_yaw);n.goal=(0.,0.,goal_yaw);n._control()
    assert n.commands[-1].angular.z==pytest.approx(sign*.04)
    assert n.settle_started is None and n.goal is not None


@pytest.mark.parametrize('fault',['odom_stale','front_stale','rear_stale','missing_scan','front_obstacle','rear_obstacle','timeout','no_goal','no_pose'])
def test_yaw_actual_control_preserves_freshness_obstacle_and_failure_stops(fault):
    n=node();n.goal=(0.,0.,.3);n.last_command.angular.z=.2
    if fault.endswith('_stale'):
        attribute={'odom_stale':'last_odom_time','front_stale':'last_front_scan_time','rear_stale':'last_rear_scan_time'}[fault]
        n.now_ns=1_000_000_001;setattr(n,attribute,Time(nanoseconds=0))
    elif fault=='missing_scan':n.last_rear_scan_time=None
    elif fault=='front_obstacle':n.front_clearance=.279
    elif fault=='rear_obstacle':n.rear_clearance=.279
    elif fault=='timeout':n.goal_started=Time(nanoseconds=0);n.now_ns=46_000_000_000;normal.advance(n,0)
    elif fault=='no_goal':n.goal=None
    else:n.pose=None
    n._control()
    assert n.commands and all(vector(c)==(0.,0.,0.) for c in n.commands)
    if fault.endswith('_stale') or fault=='missing_scan':assert n.goal is None and n.statuses[-1][1]['reason']=='stale_navigation_sensor'
    elif fault=='timeout':assert n.goal is None and n.statuses[-1][1]['reason']=='timeout'
    elif fault.endswith('_obstacle'):assert n.blocked and n.goal is not None


@pytest.mark.parametrize('event',['stop','cancel','abort'])
def test_yaw_actual_cancel_wire_prevents_further_motion(event):
    n=node();n.goal=(0.,0.,.3);n._control();assert n.commands[-1].angular.z>0
    n.commands.clear();n._on_command(NS(data=json.dumps({'event':event})))
    normal.advance(n);n._control()
    assert n.goal is None and all(vector(c)==(0.,0.,0.) for c in n.commands)
    assert n.statuses[-1][0]=='cancelled'


@pytest.mark.parametrize('gain',[1.35,1.8,2.])
def test_yaw_terminal_taper_precedes_unchanged_full_settle(gain):
    n=node(gain);n.goal=(0.,0.,0.);n.last_command.angular.z=.55
    previous=.55;first_settle=None;seen_taper=False
    for _ in range(40):
        n._control();current=n.commands[-1].angular.z
        assert abs(current-previous)<=.8/20+1e-12
        if abs(current)>.005:seen_taper=True;assert n.settle_started is None
        if n.settle_started is not None and first_settle is None:first_settle=n.settle_started.nanoseconds
        if n.goal is None:break
        previous=current;normal.advance(n)
    assert seen_taper and n.goal is None and n.statuses[-1][0]=='reached'
    assert first_settle is not None and n.now_ns-first_settle>=450_000_000


@pytest.mark.parametrize('goal',[(.046,0.,.04),(.04,0.,.046),(.04,0.,-.046)])
def test_yaw_same_position_and_heading_tolerances_still_gate_settling(goal):
    n=node();n.goal=goal;n._control()
    assert n.goal==goal and n.settle_started is None


@pytest.mark.parametrize('side',['front','rear'])
def test_yaw_obstacle_during_taper_stops_without_reached(side):
    n=node();n.goal=(0.,0.,0.);n.last_command.angular.z=.3;setattr(n,side+'_clearance',.279)
    n._control()
    assert n.blocked and n.goal is not None and n.settle_started is None
    assert vector(n.commands[-1])==(0.,0.,0.) and all(event!='reached' for event,_ in n.statuses)


@pytest.mark.parametrize('target',[-math.pi/3,math.pi/3,math.pi-.1])
def test_yaw_ideal_command_following_reaches_same_heading_with_original_steps(target):
    # Exact command-following synthetic odometry exercises actual control ordering only.
    n=node();n.goal=(0.,0.,target);previous=0.;first_settle=None
    for _ in range(250):
        n._control();command=n.commands[-1]
        assert command.linear.x==command.linear.y==0. and abs(command.angular.z)<=.55
        assert abs(command.angular.z-previous)<=.8/20+1e-12
        if n.settle_started is not None and first_settle is None:first_settle=n.settle_started.nanoseconds
        if n.goal is None:break
        n.pose=(0.,0.,nav.normalize_angle(n.pose[2]+command.angular.z/20));previous=command.angular.z;normal.advance(n)
    assert n.goal is None and n.statuses[-1][0]=='reached'
    assert abs(nav.normalize_angle(target-n.pose[2]))<=.045
    assert first_settle is not None and n.now_ns-first_settle>=450_000_000
