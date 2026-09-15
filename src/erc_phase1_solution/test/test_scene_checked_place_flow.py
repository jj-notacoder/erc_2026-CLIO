"""AST-executed actual PLACE body and pure planning orchestration; no ROS."""
from pathlib import Path
import ast
import importlib
import sys
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
from erc_phase1_solution import scene_checked_place as mod
from erc_phase1_solution.settled_torso import choose_measured_place_height, require_planned_place_height
from erc_phase1_solution.book_centered_place import book_centered_place_target
from erc_phase1_solution.motion_profiles import HOME
from erc_phase1_solution.place_contact_guard import activate, require_clear


def method(tree, name):
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'ManipulationNode')
    return next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)


def place_function():
    tree = ast.parse((HERE/'erc_phase1_solution/manipulation_node.py').read_text())
    fn = method(tree, '_place')
    from erc_phase1_solution.raised_place_finish import normal_finish
    namespace = dict(raised_place_normal_finish=normal_finish,__package__="erc_phase1_solution",np=np, book_centered_place_target=book_centered_place_target,
                     HOME=HOME, get_package_share_directory=lambda p:p,
                     plan_scene_checked_place=None,
                     plan_node_scene_checked_place=__import__("erc_phase1_solution.place_geometry_backend",fromlist=["plan_node_scene_checked_place"]).plan_node_scene_checked_place,
                     choose_measured_place_height=choose_measured_place_height,
                     require_planned_place_height=require_planned_place_height,
                     _activate_place_contacts=activate, _require_place_contact_clear=require_clear)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'candidate_place','exec'),namespace)
    return namespace


class PlaceWiring(unittest.TestCase):
    def setUp(self):
        from itertools import product
        corners = np.asarray(list(product((-.08,.08),(-.01,.01),(-.125,.125))))
        self.node = types.SimpleNamespace(
            _held_book_corners=corners,
            _carried_staging_solution=np.asarray([.35,.4,.5,.6,.7,.8,.9,1.]),
            _gravity_supported_payload=True, _cancel=threading.Event(),
            book_centered_place_enabled=True, book_centered_place_height_above_point=.27,
            book_centered_place_approach_x=.60, place_torso_height=.30,
            place_joint_limit_margin=.03, cartesian_step=.06,
        )
        n=self.node
        self.calls=[]
        n._fresh_retention_probe=lambda *a,**k:self.calls.append(('retention',a)) or True
        n._measured_left_solution=lambda:np.asarray([.35,1.,1.,1.,1.,1.,1.,1.])
        n._wait_for_perception_point=lambda key:np.asarray([.868,.004,.751])
        n._interpolate_positions=lambda a,b,step:[b.copy()]
        n._supported_compact_goals=lambda *a:(_ for _ in ()).throw(AssertionError('old prefix called'))
        n._publish_status=lambda *a,**k:self.calls.append(('status',k))
        n._best_effort=lambda fn:fn()
        n._move_torso=lambda *a,**kw:self.calls.append(('torso',a)) or True
        n._execute_retained_arm_legs=lambda legs,*a:self.calls.append(('execute',legs)) or (True,len(legs),False)
        n._open_gripper=lambda **kw:self.calls.append(('open',kw)) or True
        n._return_from_bin=lambda *a:self.calls.append(('return',a)) or True
        self.namespace=place_function()
        self.plan_calls=[]
        def planner(node,positions,rotation,carried,torso,seed,point,resolver,table_scene=None):
            self.plan_calls.append((positions,rotation,carried,torso,seed,point))
            q0=torso.copy();q0[1]=.6
            q1=q0.copy();q1[1]=.7
            setup=torso.copy();setup[1]=.5
            return mod.SceneCheckedPlacePlan([q0,q1],0,1.,[setup,q0],[torso.copy()],{'checked':True})
        self.namespace['plan_scene_checked_place']=planner

    def test_no_reverse_prefix_and_checked_setup_return_are_the_dispatched_routes(self):
        self.assertTrue(self.namespace['_place'](self.node))
        self.assertEqual(len(self.plan_calls),1)
        _,_,actual,torso,seed,_=self.plan_calls[0]
        self.assertEqual(actual[1],1.)
        self.assertEqual(seed[1],.4)
        self.assertEqual(torso[0],.30)
        kinds=[c[0] for c in self.calls]
        self.assertLess(kinds.index('status'),kinds.index('torso'))
        self.assertLess(kinds.index('execute'),kinds.index('open'))
        legs=next(v for k,v in self.calls if k=='execute')
        self.assertEqual([x[0][1] for x in legs],[.5,.6,.7])
        back=next(v for k,v in self.calls if k=='return')
        self.assertEqual([q[1] for q in back[1]],[.5])
        self.assertEqual(back[2][1],1.)

    def test_rejected_scene_cannot_reach_torso_release_or_return(self):
        def reject(*args,**kwargs):raise RuntimeError('scene rejected')
        self.namespace['plan_scene_checked_place']=reject
        with self.assertRaisesRegex(RuntimeError,'scene rejected'):
            self.namespace['_place'](self.node)
        self.assertEqual([c[0] for c in self.calls],['retention'])

    def test_postplanning_retention_failure_blocks_motion(self):
        self.node._fresh_retention_probe=lambda command,phase,**kw:phase=='post_navigation'
        with self.assertRaisesRegex(RuntimeError,'before placement'):
            self.namespace['_place'](self.node)
        self.assertNotIn('torso',[c[0] for c in self.calls])

    def test_release_correlation_still_measures_open_before_checked_return(self):
        self.node.delivery_evidence_enabled=True
        self.node._target_book_model='target'
        self.namespace['AttemptIdentity']=lambda **kwargs:types.SimpleNamespace(**kwargs)
        def measure(node,identity,goal,event):
            self.calls.append(('measurement',event));return {'verified':True}
        self.namespace['observe_measured_open_pose']=measure
        self.node._open_gripper=lambda **kw:kw['verify_measurement']()
        self.assertTrue(self.namespace['_place'](self.node,{
            'trial_id':'trial','placement_attempt_id':'attempt','target_model':'target'}))
        self.assertLess(self.calls.index(('measurement','placement_open_measured')),
                        next(i for i,c in enumerate(self.calls) if c[0]=='return'))
        self.assertEqual(self.calls[-1],('measurement','placement_hand_return_measured'))

    def test_unverified_measured_release_never_enters_return(self):
        self.node.delivery_evidence_enabled=True;self.node._target_book_model='target'
        self.namespace['AttemptIdentity']=lambda **kwargs:types.SimpleNamespace(**kwargs)
        self.namespace['observe_measured_open_pose']=lambda *a:{'verified':False}
        self.node._open_gripper=lambda **kw:kw['verify_measurement']()
        with self.assertRaisesRegex(RuntimeError,'placement_release_unverified'):
            self.namespace['_place'](self.node,{
                'trial_id':'trial','placement_attempt_id':'attempt','target_model':'target'})
        self.assertNotIn('return',[c[0] for c in self.calls])
