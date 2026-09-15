"""Actual-class differential tests on small controlled fixtures; no ROS/assets.

Most predicates are read-only spies, so equivalence covers their order and exact
arguments rather than asserting that a stub certifies real model geometry.
Two small screen controls use the existing actual box/triangle predicate.
The later moving-axis certificate is explicitly forced to fall through in this
bounds-only differential; its real behavior has a separate focused suite.
"""
from __future__ import annotations
import ast
import copy
import itertools
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
# Exact production types let custom fixtures retain the original fallback.
from erc_phase1_solution.scene_checked_place import (
    NominalBinObstacle, TableSceneObstacle, ScreenEnvelope, ShelfCradleGeometry,
)

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
BASE = HERE/'fixtures/scene_world_bounds_original.py'
NEW = PACKAGE/'erc_phase1_solution/scene_checked_place.py'
KIN = PACKAGE/'erc_phase1_solution/kinematics.py'
PALM = 'gripper_left_base_link'

def extract(path, names, scope):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    body=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in body}==set(names)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*body],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
    return scope

GEOM=extract(KIN,['_box_triangles','oriented_box_from_corners','oriented_box_intersects_triangles'],{'np':np})
TRI=(((GEOM['_box_triangles']([2.,2.,2.])+1.)/2)@np.array([4.,2.,1.])).astype(int)

def corners(low,high):
    return np.array(list(itertools.product(*zip(low,high))),dtype=float)

def box(low=(-.1,-.1,.9),high=(.1,.1,1.1)):
    return corners(low,high)[TRI].copy()

def signature(value):
    a=np.asarray(value)
    return (a.shape,a.dtype.str,a.tobytes(order='C'))

class ReductionSites(ast.NodeTransformer):
    """Instrument only original axis=(0,1) reduction sites, preserving arrays."""
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if not any(k.arg=='axis' and isinstance(k.value,ast.Tuple)
                   and [getattr(x,'value',None) for x in k.value.elts]==[0,1] for k in node.keywords):
            return node
        if isinstance(node.func,ast.Attribute) and node.func.attr in ('min','max'):
            args=[node.func.value,ast.Constant(node.func.attr)]
        elif (isinstance(node.func,ast.Call) and isinstance(node.func.func,ast.Name)
              and node.func.func.id=='getattr' and len(node.func.args)==2):
            args=node.func.args
        else:return node
        return ast.copy_location(ast.Call(func=ast.Name(id='_reduce_site',ctx=ast.Load()),args=args,keywords=[]),node)

def classes(path,env,instrument=False):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    body=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))
          and n.name in ('_finite','_SampleWorldAabbs','PlaceSceneChecker')]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*body],type_ignores=[])
    if instrument:module=ReductionSites().visit(module)
    scope=dict(np=np,math=math,PALM_COLLISION_LINK=PALM,_BOX_CORNER_TRIANGLES=TRI,
               oriented_box_intersects_triangles=env.screen_predicate,
               triangle_meshes_intersect=env.pair_predicate,PreparedTriangleMesh=env.prepare,
               _reduce_site=env.reduce_site,separated_on_axes=env.separation_fallback,
               NominalBinObstacle=NominalBinObstacle, TableSceneObstacle=TableSceneObstacle,
               ScreenEnvelope=ScreenEnvelope, ShelfCradleGeometry=ShelfCradleGeometry)
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
    return scope

class CustomArray(np.ndarray):
    pass

class Fixture:
    def __init__(self,*,screen=True,loaded=True,compound=False,subclass=False,
                 robot=None,tool=None,fail=None,real_screen=False,mutate_screen=False):
        self.trace=[];self.reductions=[];self.separation_requests=0;self.fail=fail;self.real_screen=real_screen
        self.mutate_screen=mutate_screen;self.loaded=loaded
        self.right=np.zeros(7);self.head=np.zeros(2);self.cancelled=False
        self.robot = [(name,a.copy()) for name,a in (robot or [('arm_left_3_link',box())])]
        self.tools = {name:a.copy() for name,a in (tool or [('tip',box((-.05,-.05,.95),(.05,.05,1.05)))])}
        if compound:self.robot.append((self.robot[0][0],self.robot[0][1].copy()+[.02,0,0]))
        if subclass:
            self.robot=[(n,a.view(CustomArray)) for n,a in self.robot]
            self.tools={n:a.view(CustomArray) for n,a in self.tools.items()}
        self.meshes=[NS(link=n,triangles=a,watertight=True) for n,a in self.robot]
        self.obstacle=self.observer('bin');self.table=self.observer('table')
        self.screen = NS(world=self.screen_world) if screen else None
        self.screen_bounds=((-2.,-2.,.5),(2.,2.,1.5))
        self.node=NS(_cancel=NS(is_set=lambda:self.cancelled),_lock=None,
            carried_collision_meshes=self.meshes,
            _resolved_right_positions=lambda _:self.right.copy(),
            _resolved_head_positions=lambda _:self.head.copy(),
            _collision_link_transforms=self.transforms,chain=NS(forward=self.hand),
            _robot_self_collision=self.body,_watertight_collision_links=self.closed,
            carried_transition_samples=3)
        self.tool=NS(local_surfaces=self.local_tool,watertight={k:True for k in self.tools})
        self.attached=corners((-.02,-.02,1.4),(.02,.02,1.5))

    def observer(self,name):
        owner=self
        class Observer:
            last_intersection='fixture_solid'
            def intersects(self,surface,transform,watertight):
                return owner.call(name,signature(surface),signature(transform),watertight)
            def book_intersects(self,book):return owner.call('book_bin',signature(book))
            def intersects_box(self,book):return owner.call('book_table',signature(book))
        return Observer()

    def call(self,name,*args):
        self.trace.append((name,*args))
        return name==self.fail

    def transforms(self,q,**context):
        self.trace.append(('transforms',signature(q),signature(context['right_positions']),signature(context['head_positions'])))
        t=np.eye(4);t[:3,3]=[q[1]*.01,self.right[0]*.01,self.head[0]*.01]
        return {n:t.copy() for n,_ in self.robot}

    def hand(self,q):
        self.trace.append(('hand',signature(q)))
        t=np.eye(4);t[0,3]=q[2]*.01
        return t

    def local_tool(self,aperture):
        self.trace.append(('tool_surfaces',aperture))
        return self.tools

    def screen_world(self,torso,head):
        self.trace.append(('screen_world',torso,signature(head)))
        return corners(*self.screen_bounds)

    def screen_predicate(self,box_corners,surface,**kw):
        before=signature(surface)
        self.trace.append(('screen_predicate',signature(box_corners),before,kw))
        value=(GEOM['oriented_box_intersects_triangles'](box_corners,surface,**kw)
               if self.real_screen else self.fail=='screen_predicate')
        if self.real_screen:assert signature(surface)==before
        if self.mutate_screen:surface[...,2]+=.003
        return value

    def body(self,q,**context):
        self.trace.append(('body',signature(q),signature(context['right_positions']),signature(context['head_positions'])))
        return ('body_a','body_b') if self.fail=='body' else None

    def closed(self):
        self.trace.append(('closed_links',))
        return {n for n,_ in self.robot}

    def separation_fallback(self,*args):
        self.separation_requests+=1
        return False

    def prepare(self,surface):
        self.trace.append(('prepare',signature(surface)))
        a=np.asarray(surface,dtype=float)
        if not np.isfinite(a).all():raise ValueError('collision surface coordinates must be finite')
        return NS(surface=np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape))

    def pair_predicate(self,first,second,**kw):
        self.trace.append(('pair',signature(first.surface),signature(second.surface),kw))
        if self.fail=='pair':return True
        if self.fail=='actual_box_pair':
            low=first.surface.min(axis=(0,1));high=first.surface.max(axis=(0,1))
            return GEOM['oriented_box_intersects_triangles'](corners(low,high),second.surface,
                                                            closed_surface=kw['second_watertight'])
        return False

    def reduce_site(self,surface,operation):
        value=getattr(surface,operation)(axis=(0,1))
        self.reductions.append((operation,signature(surface),signature(value)))
        return value

    def checker(self,path,instrument=False):
        return classes(path,self,instrument)['PlaceSceneChecker'](self.node,self.obstacle,self.tool,
                                                               self.attached,self.table,self.screen)

Q=np.array([.35,0.,0.,0.,0.,0.,0.,0.])

def outcome(checker,q=Q,aperture=.017,loaded=True):
    try:r=('return',checker.sample(q,aperture,loaded))
    except Exception as e:r=('raise',type(e).__name__,str(e))
    return (r,copy.deepcopy(checker.last_rejection),checker.minimum_moving_left_z,checker.samples,checker.cache_hits)

class Differential(unittest.TestCase):
    def compare(self,**kwargs):
        a,b=Fixture(**kwargs),Fixture(**kwargs)
        ca,cb=a.checker(BASE,True),b.checker(NEW,True)
        ra,rb=outcome(ca,loaded=a.loaded),outcome(cb,loaded=b.loaded)
        self.assertEqual(ra,rb);self.assertEqual(a.trace,b.trace)
        # Every cached result used later must equal the same original reduction,
        # including signed zeros/NaN payload bytes. New code may repeat less.
        self.assertTrue(all(v in a.reductions for v in b.reductions))
        return a,b,ca,cb,ra

    def test_clear_all_predicates_and_reduction_saving(self):
        a,b,_,_,r=self.compare()
        self.assertEqual(r[0],('return',True))
        self.assertLess(len(b.reductions),len(a.reductions))
        self.assertTrue(any(t[0]=='pair' for t in a.trace))
        self.assertEqual(a.separation_requests,0)
        self.assertGreater(b.separation_requests,0)

    def test_compound_same_link_exact_concatenation_order(self):self.compare(compound=True)
    def test_screen_absent(self):self.compare(screen=False)
    def test_unloaded_does_not_test_book(self):
        a,*_=self.compare(loaded=False)
        self.assertFalse(any(t[0] in ('book_bin','book_table') for t in a.trace))

    def test_each_early_collision_preserves_first_reject_and_order(self):
        expected={'bin':'robot_bin','table':'robot_table','screen_predicate':'head_screen',
                  'book_bin':'book_bin','book_table':'book_table','body':'robot_self','pair':'tool_robot'}
        for fail,reason in expected.items():
            with self.subTest(fail=fail):
                *_,r=self.compare(fail=fail)
                self.assertEqual(r[1]['reason'],reason)

    def test_tool_specific_bin_and_table_rejections(self):
        for fail in ('bin','table'):
            with self.subTest(fail=fail):
                fixtures=[]
                for path in (BASE,NEW):
                    f=Fixture();original=f.call
                    def call(name,*args,_f=f,_original=original):
                        value=_original(name,*args)
                        return value or (name==fail and args[0]==signature(_f.tools['tip']))
                    f.call=call;c=f.checker(path);fixtures.append((outcome(c),f.trace))
                self.assertEqual(*fixtures)
                self.assertEqual(fixtures[0][0][1]['reason'],'tool_'+fail)

    def test_book_screen_first_reject_after_robot_tool_checks(self):
        fixtures=[]
        for path in (BASE,NEW):
            f=Fixture();old=f.screen_predicate
            def predicate(corners_arg,surface,_f=f,_old=old,**kw):
                _old(corners_arg,surface,**kw)
                return signature(corners_arg)==signature(_f.attached)
            f.screen_predicate=predicate;c=f.checker(path);fixtures.append((outcome(c),f.trace))
        self.assertEqual(*fixtures);self.assertEqual(fixtures[0][0][1]['reason'],'book_head_screen')

    def test_left_arm_ground(self):
        *_,r=self.compare(screen=False,robot=[('arm_left_3_link',box((0,0,0),(.1,.1,.1)))])
        self.assertEqual(r[1]['reason'],'left_arm_ground')

    def test_tool_ground(self):
        *_,r=self.compare(screen=False,tool=[('tip',box((0,0,0),(.1,.1,.1)))])
        self.assertEqual(r[1]['reason'],'tool_ground')

    def test_palm_wrist_adjacency_preserved(self):
        a,_,_,_,r=self.compare(robot=[('arm_left_7_link',box())],tool=[(PALM,box())],fail='pair')
        self.assertEqual(r[0],('return',True));self.assertFalse(any(t[0]=='pair' for t in a.trace))

    def test_screen_touch_actual_predicate(self):
        fixtures=[]
        for path in (BASE,NEW):
            f=Fixture(real_screen=True);f.screen_bounds=((.1,-.1,.9),(.2,.1,1.1));c=f.checker(path)
            fixtures.append((outcome(c),f.trace))
        self.assertEqual(*fixtures);self.assertEqual(fixtures[0][0][1]['reason'],'head_screen')

    def test_screen_contained_by_closed_mesh_actual_predicate(self):
        fixtures=[]
        for path in (BASE,NEW):
            f=Fixture(real_screen=True);f.screen_bounds=((-0.01,-.01,.99),(.01,.01,1.01));c=f.checker(path)
            fixtures.append((outcome(c),f.trace))
        self.assertEqual(*fixtures);self.assertEqual(fixtures[0][0][1]['reason'],'head_screen')

    def test_tool_robot_touch_and_containment_reach_positive_predicate(self):
        for surface in (box((.1,-.05,.95),(.2,.05,1.05)),box((-.01,-.01,.99),(.01,.01,1.01))):
            *_,r=self.compare(screen=False,tool=[('tip',surface)],fail='actual_box_pair')
            self.assertEqual(r[1]['reason'],'tool_robot')

    def test_custom_mutating_screen_surface_uses_original_fallback(self):self.compare(subclass=True,mutate_screen=True)

    def test_nonfinite_and_empty_surfaces_same_error_boundary(self):
        for bad in (np.empty((0,3,3)),np.full((1,3,3),np.nan),np.full((1,3,3),np.inf)):
            with self.subTest(shape=bad.shape):self.compare(robot=[('arm_left_3_link',bad)])

    def test_cancel_precedes_cached_verdict(self):
        for path in (BASE,NEW):
            f=Fixture();c=f.checker(path);self.assertTrue(c.sample(Q,.017,True));f.cancelled=True
            self.assertFalse(c.sample(Q,.017,True));self.assertEqual(c.last_rejection,{'reason':'cancelled'})

    def test_invalid_q_and_aperture_same_errors(self):
        for q,ap in (([0]*7,.017),([math.nan]*8,.017),(Q,math.inf),(Q,math.nan)):
            a,b=Fixture(),Fixture();self.assertEqual(outcome(a.checker(BASE),q,ap),outcome(b.checker(NEW),q,ap))

    def test_multiple_q_aperture_loaded_and_parked_contexts(self):
        a,b=Fixture(),Fixture();ca,cb=a.checker(BASE,True),b.checker(NEW,True)
        for i in range(6):
            for f in (a,b):f.right[0]=i*.01;f.head[1]=i*.02
            q=Q.copy();q[1]=i*.03
            self.assertEqual(outcome(ca,q,.017+i*.005,i%2==0),outcome(cb,q,.017+i*.005,i%2==0))
        self.assertEqual(a.trace,b.trace);self.assertLess(len(b.reductions),len(a.reductions))

    def test_exact_repeat_cache_and_source_mutation_with_changed_key(self):
        a,b=Fixture(),Fixture();ca,cb=a.checker(BASE,True),b.checker(NEW,True)
        self.assertEqual(outcome(ca),outcome(cb));na,nb=len(a.trace),len(b.trace)
        self.assertEqual(outcome(ca),outcome(cb));self.assertEqual((len(a.trace),len(b.trace)),(na,nb))
        for f in (a,b):f.meshes[0].triangles[...,0]+=.025
        q=Q.copy();q[1]=.01
        self.assertEqual(outcome(ca,q),outcome(cb,q));self.assertEqual(a.trace,b.trace)

    def test_instrumentation_does_not_change_verdict_or_predicate_bytes(self):
        for path in (BASE,NEW):
            a,b=Fixture(compound=True),Fixture(compound=True)
            self.assertEqual(outcome(a.checker(path,False)),outcome(b.checker(path,True)))
            self.assertEqual(a.trace,b.trace)

    def test_readonly_sources_and_flags_unchanged(self):
        for path in (BASE,NEW):
            f=Fixture();arrays=[a for _,a in f.robot]+list(f.tools.values())
            for a in arrays:a.flags.writeable=False
            before=[(signature(a),a.flags.writeable) for a in arrays]
            self.assertEqual(outcome(f.checker(path))[0],('return',True))
            self.assertEqual(before,[(signature(a),a.flags.writeable) for a in arrays])

class HelperOwnership(unittest.TestCase):
    def setup_helper(self):
        f=Fixture();scope=classes(NEW,f,True)
        return f,scope['_SampleWorldAabbs']()

    def test_lazy_exact_bytes_signed_zero_duplicates(self):
        f,h=self.setup_helper();source=np.array([[[0.,-0.,1.],[0.,-0.,1.],[1.,2.,3.]]]);t=np.eye(4)
        w=source@t[:3,:3].T+t[:3,3];h.remember_transform(w,source,t)
        self.assertEqual(f.reductions,[])
        for op in ('max','max','min','min'):
            self.assertEqual(signature(h.reduce(w,op)),signature(getattr(w,op)(axis=(0,1))))
        self.assertEqual([r[0] for r in f.reductions],['max','min'])

    def test_mutable_sources_cannot_change_owned_world(self):
        f,h=self.setup_helper();source=box();t=np.eye(4);w=source@t[:3,:3].T+t[:3,3]
        before=signature(w);flags=(source.flags.writeable,t.flags.writeable,w.flags.writeable)
        h.remember_transform(w,source,t);h.reduce(w,'min');source+=7;t[:3,3]=8
        self.assertEqual(signature(w),before);self.assertEqual(signature(h.reduce(w,'max')),signature(w.max(axis=(0,1))))
        self.assertEqual(flags,(source.flags.writeable,t.flags.writeable,w.flags.writeable))

    def test_aliased_world_view_observes_between_use_mutation(self):
        f,h=self.setup_helper();source=box();t=np.eye(4);w=source.view()
        h.remember_transform(w,source,t);first=h.reduce(w,'min').copy();w+=1
        self.assertTrue(np.array_equal(h.reduce(w,'min'),first+1));self.assertEqual(len(f.reductions),2)

    def test_subclass_and_non_float_provenance_fallback(self):
        for kind in ('source_subclass','transform_subclass','source_float32','transform_float32','output_subclass'):
            f,h=self.setup_helper();source=box();t=np.eye(4)
            if kind=='source_subclass':source=source.view(CustomArray)
            if kind=='transform_subclass':t=t.view(CustomArray)
            if kind=='source_float32':source=source.astype(np.float32)
            if kind=='transform_float32':t=t.astype(np.float32)
            w=source@t[:3,:3].T+t[:3,3]
            if kind=='output_subclass':w=w.view(CustomArray)
            h.remember_transform(w,source,t);h.reduce(w,'max');w+=.125;h.reduce(w,'max')
            self.assertEqual(len(f.reductions),2,kind)

    def test_compound_with_ineligible_component_not_cached(self):
        f,h=self.setup_helper();source=box();t=np.eye(4);a=source@t[:3,:3].T+t[:3,3]
        h.remember_transform(a,source,t);b=a.view();merged=h.merge([a,b]);h.reduce(merged,'min');merged+=1;h.reduce(merged,'min')
        self.assertEqual(len(f.reductions),2)

    def test_compound_all_private_exact_order_and_lazy_bounds(self):
        f,h=self.setup_helper();source=box();t=np.eye(4);parts=[]
        for x in (0.,1.):
            t=t.copy();t[0,3]=x;w=source@t[:3,:3].T+t[:3,3];h.remember_transform(w,source,t);parts.append(w)
        w=h.merge(parts);self.assertEqual(signature(w),signature(np.concatenate(parts)))
        for _ in range(2):h.reduce(w,'min');h.reduce(w,'max')
        self.assertEqual(len(f.reductions),2)

    def test_empty_and_bad_shape_use_original_reductions(self):
        for w in (np.empty((0,3,3)),np.zeros((3,3)),np.zeros((1,2,3))):
            f,h=self.setup_helper();h.remember_transform(w,box(),np.eye(4))
            def result(function):
                try:return ('value',signature(function()))
                except Exception as e:return ('error',type(e).__name__,str(e))
            self.assertEqual(result(lambda:h.reduce(w,'min')),result(lambda:w.min(axis=(0,1))))

if __name__=='__main__':unittest.main(verbosity=2)
