"""Real helper/method AST tests, portable without ROS or model replay."""
import ast
from pathlib import Path
from types import SimpleNamespace
import threading
import tempfile
import unittest
from unittest.mock import patch
import sys
import types
import xml.etree.ElementTree as ET

import math
import numpy as np
from erc_phase1_solution.raised_place_finish import checked_enabled as checked_raised_place_finish_enabled

SOURCE=Path(__file__).resolve().parents[1]/'erc_phase1_solution'
IK=('torso_lift_joint',)+tuple(f'arm_left_{i}_joint' for i in range(1,8))
RIGHT=tuple(f'arm_right_{i}_joint' for i in range(1,8))
HEAD=('head_1_joint','head_2_joint')
MASTER='gripper_left_finger_joint'
NAMES=(*IK,*RIGHT,*HEAD,MASTER)


def load_helper(timer):
    tree=ast.parse((SOURCE/'empty_pickup_collision.py').read_text())
    functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('_fixed','measured_context','joint_velocity_limits','wait_for_geometry_endpoint')]
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='EmptyPickupCollision')
    methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ('require_fresh','wait_for_endpoint')]
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*functions,
          ast.ClassDef(name='Guard',bases=[],keywords=[],decorator_list=[],body=methods)]
    scope=dict(np=np,math=math,time=timer,ET=ET,IK_JOINTS=IK,RIGHT_ARM_JOINTS=RIGHT,HEAD=HEAD,MASTER=MASTER,NAMES=NAMES)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'real_empty_helper','exec'),scope)
    return scope


class Fixture:
    def __init__(self,records=None):
        self.now=1_000_000_000;self.wall=0.;self.records=list(records or [])
        self.events=[];self.on_sleep=None
        self.timer=SimpleNamespace(monotonic=lambda:self.wall,sleep=self.sleep)
        self.scope=load_helper(self.timer)
        self.node=SimpleNamespace(_lock=threading.Lock(),_cancel=threading.Event(),timeout=120.,
            joints={n:0. for n in NAMES},_joint_stamps_ns={n:self.now-1 for n in NAMES},
            adaptive_endpoint_tolerance=.0006,
            get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=self.now)),
            _publish_status=lambda event,**fields:self.events.append(dict(event=event,**fields)))
        self.node.joints[MASTER]=.069
        self.guard=self.scope['Guard']();self.guard.node=self.node
        self.guard.right=np.zeros(7);self.guard.head=np.zeros(2)
        self.guard.velocity_limits=np.asarray([.035,*([2.]*7)])
    def sleep(self,duration):
        if self.records:
            now,q=self.records.pop(0)
            self.wall+=(now-self.now)/1e9*2.4;self.now=now
            for n,v in zip(IK,q):self.node.joints[n]=float(v)
        else:
            self.now+=int(duration*1e9);self.wall+=duration*2.4
        self.node._joint_stamps_ns={n:self.now-1 for n in NAMES}
        if self.on_sleep:self.on_sleep()
    def wait(self,previous,target):
        return self.guard.wait_for_endpoint(previous,target,aperture=.069,phase='test')


class EndpointWaitTests(unittest.TestCase):
    def test_recorded_torso_residual_waits_until_measured_within_same_two_mm(self):
        # Saved sparse producer samples, supplied successively without
        # interpolation. This protocol regression is not a live counterfactual.
        values=[(177460000000,.19864195987160183),(178384000000,.23098195983011224),
                (179312000000,.2634619597884256),(180264000000,.2967819597456532),
                (181164000000,.32828195970519564),(182072000000,.3499997487011538)]
        f=Fixture([(stamp+4_000_000,np.r_[q,np.zeros(7)]) for stamp,q in values[1:]])
        f.now=values[0][0]+4_000_000
        f.node.joints[IK[0]]=values[0][1]
        f.node._joint_stamps_ns={n:values[0][0] for n in NAMES}
        target=np.r_[.35,np.zeros(7)];previous=np.r_[.10,np.zeros(7)]
        with self.assertRaisesRegex(RuntimeError,'torso differs'):f.guard.require_fresh(target,.069)
        result=f.wait(previous,target)
        self.assertAlmostEqual(result[0],values[-1][1])
        self.assertEqual([e['event'] for e in f.events],['empty_pickup_endpoint_waiting','empty_pickup_endpoint_verified'])
        self.assertGreater(f.events[-1]['ros_wait_budget_seconds'],7.14)
        self.assertEqual(f.events[-1]['joint_tolerances'],[.002,*([.008]*7)])

    def test_arm_action_success_cannot_skip_eight_milliradian_measurement(self):
        target=np.zeros(8);target[2]=1.
        f=Fixture([(1_020_000_000,np.r_[0,0,.995,np.zeros(5)])]);f.node.joints[IK[2]]=.989
        result=f.wait(np.zeros(8),target)
        self.assertEqual(result[2],.995)
        self.assertEqual(len(f.events),2)

    def test_at_target_requires_new_producer_sample_after_action_return(self):
        f=Fixture();result=f.wait(np.zeros(8),np.zeros(8))
        self.assertEqual(len(f.events),2)
        self.assertGreater(f.events[-1]['last_evaluated_measurement']['producer_stamps_ns'][IK[0]],1_000_000_000)

    def test_parked_right_head_gripper_and_uncommanded_left_fault_immediately(self):
        for name,delta in ((RIGHT[0],.0031),(HEAD[1],.0031),(MASTER,.00061),(IK[3],.0081)):
            with self.subTest(name=name):
                f=Fixture();f.node.joints[name]+=delta;target=np.r_[.35,np.zeros(7)]
                with self.assertRaises(RuntimeError):f.wait(np.zeros(8),target)
                self.assertEqual(f.wall,0.)
                self.assertEqual(f.events[-1]['event'],'empty_pickup_endpoint_rejected')

    def test_stale_or_nonfinite_measurement_is_not_waited_away(self):
        for mode in ('stale','nonfinite'):
            f=Fixture()
            if mode=='stale':f.node._joint_stamps_ns[RIGHT[2]]=0
            else:f.node.joints[IK[0]]=math.nan
            with self.assertRaisesRegex(RuntimeError,'stale'):f.wait(np.zeros(8),np.ones(8)*.1)
            self.assertEqual(f.wall,0.)

    def test_ros_deadline_is_original_and_not_extended_by_fresh_wrong_samples(self):
        f=Fixture();target=np.zeros(8);target[1]=.1
        with self.assertRaisesRegex(RuntimeError,'timed out'):f.wait(np.zeros(8),target)
        self.assertLess(f.wall,6.)
        self.assertEqual(f.events[-1]['event'],'empty_pickup_endpoint_rejected')

    def test_wall_timeout_even_if_ros_clock_stops(self):
        f=Fixture();f.node.timeout=.08
        f.timer.sleep=lambda _:setattr(f,'wall',f.wall+.03)
        with self.assertRaisesRegex(RuntimeError,'timed out'):f.wait(np.zeros(8),np.zeros(8))
        self.assertLess(f.wall,.11)

    def test_cancel_or_reversed_clock_halts_without_endpoint_acceptance(self):
        for mode in ('cancel','clock'):
            f=Fixture()
            f.on_sleep=(lambda:f.node._cancel.set()) if mode=='cancel' else lambda:setattr(f,'now',0)
            with self.assertRaises(RuntimeError):f.wait(np.zeros(8),np.zeros(8))
            self.assertNotEqual(f.events[-1]['event'],'empty_pickup_endpoint_verified')

    def test_official_velocity_limits_required_and_ordered(self):
        f=Fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'robot.urdf'
            joints=''.join(f'<joint name="{n}"><limit velocity="{.035 if i==0 else i}"/></joint>' for i,n in enumerate(IK))
            path.write_text('<robot>'+joints+'</robot>')
            values=f.scope['joint_velocity_limits'](path)
            np.testing.assert_array_equal(values,[.035,1,2,3,4,5,6,7])
            self.assertFalse(values.flags.writeable)
            path.write_text('<robot>'+joints.replace('velocity="0.035"','velocity="nan"')+'</robot>')
            with self.assertRaises(ValueError):f.scope['joint_velocity_limits'](path)

    def test_backward_clock_step_above_original_start_rejects(self):
        f=Fixture();times=iter((1_200_000_000,1_100_000_000))
        f.on_sleep=lambda:setattr(f,'now',next(times))
        target=np.zeros(8);target[1]=.1
        with self.assertRaisesRegex(RuntimeError,'clock reversed'):f.wait(np.zeros(8),target)
        self.assertEqual(f.events[-1]['last_evaluated_measurement']['evaluated_ros_ns'],1_200_000_000)

    def test_place_context_is_checked_during_wait_and_before_success(self):
        for failure_call in (1,3):
            f=Fixture();calls=[]
            def context():
                calls.append(True)
                if len(calls)==failure_call:raise RuntimeError('placement scene contact fault')
            with self.assertRaisesRegex(RuntimeError,'scene contact fault'):
                f.scope['wait_for_geometry_endpoint'](f.node,np.zeros(8),np.zeros(8),
                    right=np.zeros(7),head=np.zeros(2),velocity_limits=f.guard.velocity_limits,
                    aperture=.069,phase='empty_return_torso_home',command='place',context_check=context)
            self.assertEqual(f.events[-1]['event'],'empty_return_endpoint_rejected')
            self.assertEqual(f.events[-1]['command'],'place')

    def test_place_torso_home_wait_exceeds_old_five_wall_evidence_window(self):
        f=Fixture([(1_900_000_000,np.r_[.25,np.zeros(7)]),
                   (3_900_000_000,np.r_[.18,np.zeros(7)]),
                   (6_400_000_000,np.r_[.10,np.zeros(7)])])
        f.node.joints[IK[0]]=.28
        result=f.scope['wait_for_geometry_endpoint'](f.node,np.r_[.35,np.zeros(7)],np.r_[.1,np.zeros(7)],
            right=np.zeros(7),head=np.zeros(2),velocity_limits=f.guard.velocity_limits,
            aperture=.069,phase='empty_return_torso_home',command='place',context_check=lambda:None)
        self.assertGreater(f.wall,5.)
        self.assertAlmostEqual(result[0],.1)
        self.assertEqual(f.events[-1]['event'],'empty_return_endpoint_verified')


def pickup_execution():
    tree=ast.parse((SOURCE/'manipulation_node.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_pick')
    start=next(i for i,x in enumerate(method.body) if isinstance(x,ast.If) and '_move_torso(self.pick_torso_height, 2.5)' in ast.unparse(x))
    stop=next(i for i in range(start+1,len(method.body)) if isinstance(method.body[i],ast.Try))
    serial=method.body[start].body
    assert ast.unparse(method.body[start].test)=='torso_overlap is None'
    torso_start=next(i for i,x in enumerate(serial) if isinstance(x,ast.If) and '_move_torso(self.pick_torso_height, 2.5)' in ast.unparse(x))
    args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in ('self','empty_setup_guard','empty_elevated','transition_waypoints','solutions','staged_empty_gripper')],kwonlyargs=[],kw_defaults=[],defaults=[])
    function=ast.FunctionDef(name='execute',args=args,body=serial[torso_start:]+method.body[start+1:stop],decorator_list=[])
    from erc_phase1_solution.empty_pickup_setup_timing import move_empty_pickup_setup
    scope=dict(move_empty_pickup_setup=move_empty_pickup_setup)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'actual_pick_execution','exec'),scope)
    return scope['execute']


class PickupWiringTests(unittest.TestCase):
    def make(self,fail=None):
        events=[];actual=np.zeros(8);elevated=np.r_[.35,np.zeros(7)]
        node=SimpleNamespace(pick_torso_height=.35,_best_effort=lambda action:action())
        def move(kind,goal):events.append(('move',kind,np.asarray(goal).copy()));return True
        node._move_torso=lambda height,duration:move('torso',elevated)
        node._move_arm_solution=lambda goal,duration:move('arm',goal)
        node._open_gripper=lambda:True
        def fresh(expected,aperture):np.testing.assert_array_equal(expected,actual)
        def wait(previous,expected,**kw):
            events.append(('wait',kw['phase'],np.asarray(expected).copy()))
            if kw['phase']==fail:raise RuntimeError('endpoint still wrong')
            actual[:]=expected
        guard=SimpleNamespace(start=np.zeros(8),open_aperture=.069,require_fresh=fresh,wait_for_endpoint=wait)
        goals=[elevated+np.r_[0,np.ones(7)*v] for v in (.1,.2,.3)]
        return node,guard,elevated,goals,events

    def test_each_executed_leg_waits_before_next_geometry_admission(self):
        node,guard,elevated,goals,events=self.make()
        pickup_execution()(node,guard,elevated,[goals[0]],[goals[1],goals[2]],False)
        self.assertEqual([e[0] for e in events],['move','wait']*4)
        self.assertEqual([e[1] for e in events if e[0]=='wait'],['empty_setup_torso','empty_setup_transition','empty_setup_clearance','empty_cartesian_approach'])

    def test_torso_or_arm_failed_measurement_prevents_next_command(self):
        for phase,expected_moves in (('empty_setup_torso',1),('empty_setup_transition',2),('empty_setup_clearance',3),('empty_cartesian_approach',4)):
            node,guard,elevated,goals,events=self.make(phase)
            with self.assertRaisesRegex(RuntimeError,'still wrong'):
                pickup_execution()(node,guard,elevated,[goals[0]],[goals[1],goals[2]],False)
            self.assertEqual(sum(e[0]=='move' for e in events),expected_moves)

    def test_legacy_no_empty_guard_adds_no_waits(self):
        node,_,elevated,goals,events=self.make()
        pickup_execution()(node,None,elevated,[goals[0]],[goals[1],goals[2]],False)
        self.assertEqual([e[0] for e in events],['move']*4)


class ReturnWiringTests(unittest.TestCase):
    def method(self):
        tree=ast.parse((SOURCE/'manipulation_node.py').read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
        method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_execute_unloaded_home')
        scope=dict(__package__='erc_phase1_solution', __name__='erc_phase1_solution.manipulation_node', np=np,HOME=np.r_[.1,np.ones(7)*.2],ARM_JOINTS=IK[1:],
            checked_raised_place_finish_enabled=checked_raised_place_finish_enabled)
        body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method]
        exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'actual_empty_return','exec'),scope)
        return scope['_execute_unloaded_home'],scope['HOME']

    def test_home_arm_measured_before_torso_and_torso_measured_before_success(self):
        method,home=self.method();events=[];measured=np.r_[.35,np.zeros(7)]
        n=SimpleNamespace(arm_client=object(),_follow=lambda *a:events.append('arm_action_success') or True,
                          _move_torso=lambda *a:events.append('torso_action_success') or True)
        def wait(previous,goal,phase):
            events.append(phase);measured[:]=goal
        self.assertTrue(method(n,[],endpoint_wait=wait,endpoint_start=measured.copy()))
        self.assertEqual(events,['arm_action_success','empty_return_arm_home','torso_action_success','empty_return_torso_home'])
        np.testing.assert_array_equal(measured,home)

    def test_arm_wait_rejection_prevents_torso_and_torso_rejection_prevents_success(self):
        for failure in ('empty_return_arm_home','empty_return_torso_home'):
            method,_=self.method();events=[]
            n=SimpleNamespace(arm_client=object(),_follow=lambda *a:events.append('arm') or True,
                              _move_torso=lambda *a:events.append('torso') or True)
            def wait(previous,goal,phase):
                if phase==failure:raise RuntimeError('endpoint not measured')
            with self.assertRaisesRegex(RuntimeError,'not measured'):
                method(n,[],endpoint_wait=wait,endpoint_start=np.r_[.35,np.zeros(7)])
            self.assertEqual(events,['arm'] if failure=='empty_return_arm_home' else ['arm','torso'])

    def test_legacy_default_keeps_original_command_sequence(self):
        method,_=self.method();events=[]
        n=SimpleNamespace(arm_client=object(),_follow=lambda *a:events.append('arm') or True,
                          _move_torso=lambda *a:events.append('torso') or True)
        self.assertTrue(method(n,[]))
        self.assertEqual(events,['arm','torso'])

    def test_selected_direct_return_wires_all_waits_and_live_context(self):
        tree=ast.parse((SOURCE/'manipulation_node.py').read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
        methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ('_return_from_bin','_execute_unloaded_home')]
        home=np.r_[.1,np.ones(7)*.2];events=[];context=object()
        from erc_phase1_solution.raised_place_finish import checked_enabled
        namespace=dict(checked_raised_place_finish_enabled=checked_enabled,
                       np=np,HOME=home,ARM_JOINTS=IK[1:],RIGHT_ARM_JOINTS=RIGHT,HEAD_JOINTS=HEAD,
                       Path=Path,__package__='erc_phase1_solution',
                       __name__='erc_phase1_solution.manipulation_node',
                       get_package_share_directory=lambda name:'/declared-official-share',
                       _require_place_contact_clear=lambda node:events.append('contact_check'),
                       measured_scene_context=lambda node,ref:events.append(('scene_check',ref)))
        body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*methods]
        exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'actual_direct_return','exec'),namespace)
        node=SimpleNamespace(table_scene_required=True,_active_place_scene_reference=context,gripper_open=.069,arm_client=object())
        node._execute_unloaded_home=types.MethodType(namespace['_execute_unloaded_home'],node)
        node._move_arm_solution=lambda goal,duration:events.append(('arm',np.asarray(goal).tolist())) or True
        node._follow=lambda *args:events.append('home_arm') or True
        node._move_torso=lambda *args:events.append('home_torso') or True
        from erc_phase1_solution import release_only_place_planning as actual_release_only
        package=types.ModuleType('erc_phase1_solution');package.__path__=[]
        package.release_only_place_planning=actual_release_only
        helper=types.ModuleType('erc_phase1_solution.empty_pickup_collision')
        helper.measured_context=lambda n:({name:(.069 if name==MASTER else 0.) for name in NAMES},{})
        helper.joint_velocity_limits=lambda path:np.r_[.035,np.ones(7)*2]
        def wait(n,first,goal,**kwargs):
            self.assertEqual(kwargs['command'],'place');self.assertEqual(kwargs['aperture'],.069)
            kwargs['context_check']()
            events.append(('wait',kwargs['phase'],np.asarray(goal).tolist()))
        helper.wait_for_geometry_endpoint=wait
        path=[np.r_[.35,np.ones(7)*v] for v in (.4,.5,.6)]
        with patch.dict(sys.modules,{'erc_phase1_solution':package,
                'erc_phase1_solution.empty_pickup_collision':helper,
                'erc_phase1_solution.release_only_place_planning':actual_release_only}):
            self.assertTrue(namespace['_return_from_bin'](node,path,[],path[0],[],direct_empty_home=[]))
        waits=[e for e in events if isinstance(e,tuple) and e[0]=='wait']
        self.assertEqual([e[1] for e in waits],['empty_return_cartesian']*2+['empty_return_arm_home','empty_return_torso_home'])
        self.assertEqual(waits[-2][2][0],.35)
        self.assertEqual(waits[-1][2],home.tolist())
        self.assertTrue(all(e[1] is context for e in events if isinstance(e,tuple) and e[0]=='scene_check'))


if __name__=='__main__':unittest.main(verbosity=2)
