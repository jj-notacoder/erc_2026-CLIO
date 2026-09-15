"""Controller-law checks only; no simulator or physical retention claim."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import Twist
from rclpy.time import Time
from erc_phase1_solution import navigation_node as nav
import test_navigation_profiles as profiles

ROOT=Path(__file__).resolve().parents[1]

def parent_source():
    record_path=ROOT/'test/fixtures/normal_linear_convergence_inverse.json'
    assert hashlib.sha256(record_path.read_bytes()).hexdigest()=='a53bc2c0f4f393cb50ce7afdd9a6679f6a786e7356217559f3a529374c26e39e'
    record=json.loads(record_path.read_bytes())
    source=(ROOT/'erc_phase1_solution/navigation_node.py').read_bytes()
    assert hashlib.sha256(source).hexdigest()==record['candidate_sha256']
    for edit in reversed(record['replacements']):
        assert source.count(edit['new'].encode())==1
        source=source.replace(edit['new'].encode(),edit['old'].encode(),1)
    assert hashlib.sha256(source).hexdigest()==record['parent_sha256']
    return source

def parent_method(name):
    owner=next(n for n in ast.parse(parent_source()).body if isinstance(n,ast.ClassDef) and n.name=='NavigationNode')
    method=copy.deepcopy(next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    scope=dict(nav.__dict__)
    exec(compile(ast.fix_missing_locations(module),'<exact parent navigation method>','exec'),scope)
    return scope[name]

def control_node(enabled=True,profile='normal'):
    n=profiles._profile_node(profile)
    n.normal_linear_convergence_enabled=enabled
    n.max_wz=.55;n.now_ns=1_000_000_000
    n.get_clock=lambda:NS(now=lambda:Time(nanoseconds=n.now_ns))
    n.pose=(0.,0.,0.);n.goal=(.13625555949727775,0.,0.)
    n.goal_id=1;n.goal_started=None;n.goal_timeout=45.;n.settle_started=None
    n.last_odom_time=Time(nanoseconds=n.now_ns)
    n.last_front_scan_time=n.last_odom_time;n.last_rear_scan_time=n.last_odom_time
    n.sensor_stale_timeout=1.;n.front_clearance=2.;n.rear_clearance=2.;n.stop_distance=.28
    n.blocked=False;n._ready_announced=True
    n._sample_executed_path=lambda force=False:None
    n.statuses=[];n.commands=[]
    n._publish_status=lambda event,**kw:n.statuses.append((event,kw))
    n.cmd_pub=NS(publish=n.commands.append,get_subscription_count=lambda:1)
    return n

def advance(n,ns=50_000_000):
    n.now_ns+=ns
    n.last_odom_time=Time(nanoseconds=n.now_ns)
    n.last_front_scan_time=n.last_odom_time;n.last_rear_scan_time=n.last_odom_time

@pytest.mark.parametrize('xy',[(0.,0.),(.04,0.),(.07,0.),(.136,.0),(.349,0.),(.35,0.),(.9,-.9)])
@pytest.mark.parametrize('flag',['missing',False])
def test_default_translation_is_exact_parent(xy,flag):
    n=profiles._profile_node('normal')
    if flag!='missing':n.normal_linear_convergence_enabled=flag
    args=(*xy,math.hypot(*xy))
    assert n._desired_translation(*args)==parent_method('_desired_translation')(n,*args)

@pytest.mark.parametrize('distance',[.045,.07,.13625555949727775,math.nextafter(.35,0.)])
def test_opt_in_removes_only_multiplier_with_same_proportional_gain(distance):
    n=control_node();x=distance/math.sqrt(2);y=-x
    result=n._desired_translation(x,y,distance)
    assert result==(.85*x,.85*y)
    old=parent_method('_desired_translation')(n,x,y,distance)
    factor=max(.2,distance/.35)
    assert old==(result[0]*factor,result[1]*factor)

@pytest.mark.parametrize('profile,distance',[('carried_retreat',.35),('carried_retreat',.046),('unknown_custom',.1),('normal',.35),('normal',2.)])
def test_other_profiles_and_far_commands_are_exact_parent(profile,distance):
    n=control_node(profile=profile)
    assert n._desired_translation(distance,0.,distance)==parent_method('_desired_translation')(n,distance,0.,distance)

def test_complete_inverse_preserves_every_other_navigation_method():
    def methods(source):
        owner=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='NavigationNode')
        return {n.name:ast.dump(n,include_attributes=False) for n in owner.body if isinstance(n,ast.FunctionDef)}
    old=methods(parent_source());new=methods((ROOT/'erc_phase1_solution/navigation_node.py').read_bytes())
    assert {name for name in old if old[name]!=new[name]}=={'__init__','_declare_parameters','_desired_translation','_control'}
    assert old['_limited_command']==new['_limited_command']
    assert old['_control_empty_arm']==new['_control_empty_arm']

def test_declared_default_false_and_actual_constructor_assignment():
    source=(ROOT/'erc_phase1_solution/navigation_node.py').read_bytes()
    owner=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='NavigationNode')
    declare=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_declare_parameters')
    declaration=next(n.value for n in declare.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='values' for t in n.targets))
    assert ast.literal_eval(declaration)['normal_linear_convergence_enabled'] is False
    init=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assignment=[n for n in init.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='normal_linear_convergence_enabled' for t in n.targets)]
    assert len(assignment)==1
    for flag in (False,True):
        n=NS(get_parameter=lambda name:NS(value=flag))
        exec(compile(ast.fix_missing_locations(ast.Module(body=assignment,type_ignores=[])),'<actual option assignment>','exec'),{'self':n})
        assert n.normal_linear_convergence_enabled is flag

@pytest.mark.parametrize('fault',['stale','obstacle','timeout','no_goal'])
def test_actual_control_keeps_stop_and_failure_gates(fault):
    n=control_node();n.last_command.linear.x=.1
    if fault=='stale':
        advance(n,1_000_000_000)
        n.last_odom_time=Time(nanoseconds=n.now_ns-1_000_000_001)
    elif fault=='obstacle':n.front_clearance=.1
    elif fault=='timeout':n.goal_started=Time(nanoseconds=0);n.now_ns=46_000_000_000;advance(n,0)
    else:n.goal=None
    n._control()
    assert n.commands and n.commands[-1].linear.x==n.commands[-1].linear.y==n.commands[-1].angular.z==0.
    if fault=='stale':assert n.statuses[-1][1]['reason']=='stale_navigation_sensor'
    elif fault=='timeout':assert n.statuses[-1][1]['reason']=='timeout'
    elif fault=='obstacle':assert n.blocked and n.statuses[-1][0]=='blocked'

def test_opt_in_terminal_stop_ramps_without_larger_command_steps_before_original_settle():
    n=control_node();n.pose=(.13625555949727775-.04,0.,0.)
    n.last_command.linear.x=.03825;n.last_command.angular.z=.05
    previous=n.last_command;seen_ramp=False
    for _ in range(20):
        n._control();current=n.commands[-1]
        assert abs(current.linear.x-previous.linear.x)<=.55/20+1e-12
        assert abs(current.angular.z-previous.angular.z)<=.8/20+1e-12
        if current.linear.x or current.angular.z:
            seen_ramp=True;assert n.settle_started is None
        if n.goal is None:break
        previous=current;advance(n)
    assert seen_ramp and n.goal is None and n.statuses[-1][0]=='reached'
    assert n.now_ns>=1_000_000_000+450_000_000

@pytest.mark.parametrize('enabled',[False,True])
def test_actual_control_preserves_position_and_yaw_tolerances(enabled):
    n=control_node(enabled);n.goal=(.04,0.,.05)
    n._control();assert n.goal is not None and n.settle_started is None
    n.goal=(.046,0.,.04);advance(n);n._control()
    assert n.goal is not None and n.settle_started is None

def test_desired_command_stopping_envelope_and_caps_are_mathematical_not_measured_braking():
    n=control_node()
    for distance in (.001,.045,.07,.136,.349,.35,.5,1.,4.):
        for angle in (0.,math.pi/4,math.pi/2,math.pi,5*math.pi/4):
            x,y=distance*math.cos(angle),distance*math.sin(angle)
            vx,vy=n._desired_translation(x,y,distance)
            assert abs(vx)<=.42 and abs(vy)<=.32
            assert (vx*vx+vy*vy)/(2*.55)<=distance+1e-12

@pytest.mark.parametrize('yaw', [0.,math.pi/4,math.pi/2])
def test_nominal_20hz_command_rollout_keeps_actual_limiter_steps_and_reaches_same_tolerance(yaw):
    # Synthetic ideal odometry follows issued commands exactly; not a robot model.
    n=control_node();n.pose=(0.,0.,yaw);n.goal=(.13625555949727775,0.,yaw)
    previous=Twist();start=n.now_ns
    for _ in range(200):
        distance=math.hypot(n.goal[0]-n.pose[0],n.goal[1]-n.pose[1])
        n._control();command=n.commands[-1]
        assert abs(command.linear.x)<=.42 and abs(command.linear.y)<=.32 and abs(command.angular.z)<=.55
        assert abs(command.linear.x-previous.linear.x)<=.55/20+1e-12
        assert abs(command.linear.y-previous.linear.y)<=.55/20+1e-12
        assert abs(command.angular.z-previous.angular.z)<=.8/20+1e-12
        assert (command.linear.x**2+command.linear.y**2)/(2*.55)<=distance+1e-12
        if n.goal is None:break
        x,y,theta=n.pose
        n.pose=(x+(math.cos(theta)*command.linear.x-math.sin(theta)*command.linear.y)/20,
                y+(math.sin(theta)*command.linear.x+math.cos(theta)*command.linear.y)/20,
                theta+command.angular.z/20)
        previous=command;advance(n)
    assert n.goal is None and n.statuses[-1][0]=='reached'
    assert math.hypot(.13625555949727775-n.pose[0],n.pose[1])<=.045
    assert (n.now_ns-start)/1e9<3.

def test_default_control_terminal_behavior_remains_exact_parent():
    traces=[]
    for function in (nav.NavigationNode._control,parent_method('_control')):
        n=control_node(False);n.pose=(.1,0.,0.);n.last_command.linear.x=.03
        function(n)
        traces.append(([(c.linear.x,c.linear.y,c.angular.z) for c in n.commands],n.settle_started.nanoseconds,n.statuses))
    assert traces[0]==traces[1]
