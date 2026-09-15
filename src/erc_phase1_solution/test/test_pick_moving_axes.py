"""Actual PICK method integration with tiny meshes; no ROS, assets or motion.

Tests are prepared but unexecuted. The real separation helper is loaded from its
pinned R43 copy. Narrow-phase spies deliberately reject, so bypass is observable;
they do not assert universal equivalence with conservative legacy false positives.
"""
import ast
from empty_pickup_snapshot_support import restore_empty_pickup_snapshot
from tool_local_ast_support import restore_tool_constructor_keyword
import copy
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT/'erc_phase1_solution/empty_pickup_collision.py'
ORIGINAL = ROOT/'test/fixtures/pick_axes_original.py'
PROOF = ROOT/'erc_phase1_solution/static_pair_separation.py'
PALM = 'gripper_left_base_link'
LINKS = (PALM, 'gripper_left_base_finger_left_link',
         'gripper_left_inner_finger_left_link', 'gripper_left_outer_finger_left_link',
         'gripper_left_fingertip_left_link', 'gripper_left_base_finger_right_link',
         'gripper_left_inner_finger_right_link', 'gripper_left_outer_finger_right_link',
         'gripper_left_fingertip_right_link')
assert hashlib.sha256(PROOF.read_bytes()).hexdigest() == '1dea6499c9f594a60cd51936d52862433730a8373efd99700b7e76ecdf7585f9'
spec = importlib.util.spec_from_file_location('pick_existing_separation', PROOF)
proof_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof_module)
real_proof = proof_module.separated_on_axes


def rotate(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c,-s,0.], [s,c,0.], [0.,0.,1.]])


def box(center=(0.,0.,0.), size=(2.,.1,.1)):
    points = np.array(list(itertools.product((-1.,1.), repeat=3)))*np.array(size)/2+center
    faces = [[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
             [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
    return points[np.asarray(faces, dtype=int)]


class Fixture:
    def __init__(self, *, gap=.1, angle=.6, link=PALM, robot_link='arm_left_3_link'):
        self.events = []
        self.projections = []
        self.q = np.zeros(8)
        self.q[1] = angle
        self.angle = angle
        self.link, self.robot_link = link, robot_link
        self.offset = np.array([0.,0.,1.])
        self.local = {name: box([20.+i*3.,0.,0.]) for i,name in enumerate(LINKS)}
        self.local[link] = box()
        self.robot = {robot_link: box([0.,.1+gap,0.]) @ rotate(angle).T+self.offset}
        self.screen_center = np.array([100.,0.,1.])
        self.screen_reject = False
        self.body_reason = None
        self.narrow_result = True
        self.proof_axes = None
        self.aperture_shift = False
        self.closed = {robot_link}
        self.watertight = {name: True for name in LINKS}
        self.node = NS(chain=NS(lower=np.full(8,-4.), upper=np.full(8,4.), forward=self.hand),
            gripper_open=.069, carried_transition_samples=61, _cancel=threading.Event(),
            _robot_self_collision=self.body, _world_collision_surfaces=self.world,
            _watertight_collision_links=lambda:self.closed, pick_torso_height=.35)
        self.geometry = NS(local_surfaces=self.surfaces, watertight=self.watertight)
        self.screen = NS(world=self.screen_world)

    def hand(self, q):
        self.events.append('hand')
        t = np.eye(4)
        t[:3,:3] = rotate(q[1])
        t[:3,3] = self.offset+t[:3,1]*q[2]
        return t

    def surfaces(self, aperture):
        self.events.append('local')
        if not self.aperture_shift:
            return self.local
        return {name: s+[0.,aperture-.017,0.] for name,s in self.local.items()}

    def body(self, q, **context):
        self.events.append('body')
        return self.body_reason

    def world(self, q, **context):
        self.events.append('world')
        return self.robot

    def screen_world(self, torso, head):
        self.events.append('screen_world')
        return np.array(list(itertools.product((-.05,.05), repeat=3)))+self.screen_center

    def screen_predicate(self, screen, surface, **kwargs):
        self.events.append('screen_check')
        return self.screen_reject

    def certificate(self, first, second, axes):
        self.events.append('proof')
        self.projections.append(tuple(x.copy() for x in (first,second,axes)))
        return real_proof(first, second, axes if self.proof_axes is None else self.proof_axes)

    def prepare(self, surface):
        self.events.append('prepare')
        return NS(surface=surface)

    def narrow(self, first, second, **kwargs):
        self.events.append(('narrow', kwargs))
        if isinstance(self.narrow_result, Exception):
            raise self.narrow_result
        return self.narrow_result

    def checker(self, path=CURRENT):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        names = {'_fixed','_bounds','EmptyPickupCollision'}
        selected = [copy.deepcopy(n) for n in tree.body
                    if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
        assert {n.name for n in selected} == names
        from erc_phase1_solution.empty_pickup_collision import ShelfCradleGeometry, ScreenEnvelope
        namespace = dict(np=np, math=math, ShelfCradleGeometry=ShelfCradleGeometry,
            ScreenEnvelope=ScreenEnvelope, IK_JOINTS=tuple(range(8)), RIGHT_ARM_JOINTS=tuple(range(7)),
            LEFT_GRIPPER_COLLISION_LINKS=LINKS, PALM_COLLISION_LINK=PALM,
            separated_on_axes=self.certificate, PreparedTriangleMesh=self.prepare,
            triangle_meshes_intersect=self.narrow, oriented_box_intersects_triangles=self.screen_predicate)
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')],level=0),
                                 *selected], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
        return namespace['EmptyPickupCollision'](self.node, self.q, np.zeros(7), np.zeros(2), .069,
                                                 geometry=self.geometry, screen=self.screen)


def test_only_import_and_pair_guard_change_original_ast():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == 'bc23a0b3fa93e6ef370b4b610fb1081a4efb91d4ad0ec264ba0053927ba0d614'
    original = ast.parse(ORIGINAL.read_text(encoding='utf-8'))
    current = ast.parse(restore_empty_pickup_snapshot(CURRENT.read_text(encoding='utf-8')))
    current = restore_tool_constructor_keyword(current)
    imports = [n for n in current.body if isinstance(n,ast.ImportFrom) and n.module=='static_pair_separation']
    assert len(imports)==1 and [(n.name,n.asname) for n in imports[0].names]==[('separated_on_axes',None)]
    current.body.remove(imports[0])
    found = []
    class RemoveProof(ast.NodeTransformer):
        def visit_If(self, node):
            if isinstance(node.test, ast.Call) and isinstance(node.test.func,ast.Name) and node.test.func.id=='separated_on_axes':
                found.append(node)
                assert len(node.body)==1 and isinstance(node.body[0],ast.Continue) and not node.orelse
                return None
            return self.generic_visit(node)
    RemoveProof().visit(current)
    assert len(found)==1
    assert ast.dump(current,include_attributes=False)==ast.dump(original,include_attributes=False)


@pytest.mark.parametrize('angle',[.37,.91,1.3])
def test_actual_world_aabb_overlap_but_rotated_meshes_separate(angle):
    f = Fixture(angle=angle)
    c = f.checker()
    assert c.sample(f.q,.069) is None
    assert f.events==['body','world','local','hand','screen_world','proof']
    a,b,axes = f.projections[0]
    assert np.all(a.max(axis=(0,1))>=b.min(axis=(0,1))) and np.all(b.max(axis=(0,1))>=a.min(axis=(0,1)))
    np.testing.assert_array_equal(axes,rotate(angle).T)
    np.testing.assert_array_equal(a,f.local[PALM]@rotate(angle).T+f.offset)
    np.testing.assert_array_equal(b,f.robot[f.robot_link])
    old=Fixture(angle=angle)
    assert old.checker(ORIGINAL).sample(old.q,.069)==f'empty_pickup_tool_robot:{PALM}:arm_left_3_link'
    assert old.events[-3:]==['prepare','prepare',('narrow',dict(first_watertight=True,second_watertight=True))]


@pytest.mark.parametrize('gap',[-.01,0.,99e-6,100e-6])
def test_touch_overlap_and_margin_retain_original_rejection(gap):
    f=Fixture(gap=gap); c=f.checker()
    assert c.sample(f.q,.069)==f'empty_pickup_tool_robot:{PALM}:arm_left_3_link'
    assert f.events[-4:]==['proof','prepare','prepare',('narrow',dict(first_watertight=True,second_watertight=True))]
    assert c.last_rejected_q==f.q.tolist() and c.last_rejected_aperture==.069


@pytest.mark.parametrize('mode',['contained','compound','aliased','nan'])
def test_complete_uncertain_geometry_cannot_bypass(mode):
    f=Fixture()
    inner=f.local[PALM]@rotate(f.angle).T+f.offset
    if mode=='contained': f.robot[f.robot_link]=box(size=(4.,4.,.4))+f.offset
    elif mode=='compound': f.robot[f.robot_link]=np.concatenate([f.robot[f.robot_link],inner])
    elif mode=='aliased': f.robot[f.robot_link]=inner
    else: f.robot[f.robot_link][0,0,0]=np.nan
    c=f.checker()
    assert c.sample(f.q,.069)==f'empty_pickup_tool_robot:{PALM}:arm_left_3_link'
    assert 'proof' in f.events and f.events[-1][0]=='narrow'


def test_invalid_axis_hint_falls_through_without_changing_old_false_result():
    f=Fixture(); f.proof_axes=np.zeros((3,3)); f.narrow_result=False
    assert f.checker().sample(f.q,.069) is None
    assert f.events[-4:]==['proof','prepare','prepare',('narrow',dict(first_watertight=True,second_watertight=True))]


@pytest.mark.parametrize('gate',['body','inventory','robot_floor','screen','tool_floor','aabb'])
def test_existing_earlier_gates_precede_proof(gate):
    f=Fixture(); expected=None
    if gate=='body': f.body_reason=('left','head'); expected="empty_pickup_robot_self:('left', 'head')"
    elif gate=='inventory': f.local.pop(LINKS[-1]); expected='empty_pickup_tool_inventory_invalid'
    elif gate=='robot_floor': f.robot[f.robot_link][...,2]-=1.; expected='empty_pickup_robot_floor:arm_left_3_link'
    elif gate=='screen': f.screen_center=f.offset; f.screen_reject=True; expected='empty_pickup_screen:arm_left_3_link'
    elif gate=='tool_floor': f.local[PALM][...,2]-=1.; expected=f'empty_pickup_tool_floor:{PALM}'
    else: f.robot[f.robot_link]+=np.array([0.,10.,0.])
    assert f.checker().sample(f.q,.069)==expected
    assert 'proof' not in f.events and 'prepare' not in f.events


def test_only_existing_palm_arm7_exclusion_is_kept():
    f=Fixture(gap=-.1,robot_link='arm_left_7_link')
    assert f.checker().sample(f.q,.069) is None
    assert 'proof' not in f.events
    f=Fixture(gap=-.1,robot_link='arm_left_7_link',link=LINKS[1])
    assert f.checker().sample(f.q,.069)==f'empty_pickup_tool_robot:{LINKS[1]}:arm_left_7_link'
    assert 'proof' in f.events


@pytest.mark.parametrize('tool_link',LINKS)
def test_each_of_nine_tool_links_keeps_collision_rejection(tool_link):
    f=Fixture(link=tool_link,gap=-.1)
    assert f.checker().sample(f.q,.069)==f'empty_pickup_tool_robot:{tool_link}:arm_left_3_link'


def test_proof_success_cannot_skip_later_robot_pair_and_first_rejection_order():
    f=Fixture()
    touching=f.local[PALM]@rotate(f.angle).T+f.offset
    f.robot.update({'arm_right_2_link':touching,'torso_lift_link':touching})
    assert f.checker().sample(f.q,.069)==f'empty_pickup_tool_robot:{PALM}:arm_right_2_link'
    assert f.events.count('proof')==2
    assert len([x for x in f.events if isinstance(x,tuple) and x[0]=='narrow'])==1


@pytest.mark.parametrize('change',['q','aperture','uncached_mesh'])
def test_current_vertices_rechecked_after_geometry_change(change):
    f=Fixture(gap=.02 if change=='aperture' else .1)
    f.aperture_shift=change=='aperture'
    c=f.checker(); assert c.sample(f.q,.017) is None
    f.events.clear()
    if change=='q':
        f.q[2]=.2
        reason=c.sample(f.q,.017)
    elif change=='aperture':
        # The legal 0.052 m aperture change crosses a 0.02 m initial gap.
        reason=c.sample(f.q,.069)
    else:
        f.robot[f.robot_link]-=rotate(f.angle)[:,1]*.2
        reason=c._sample_uncached(f.q,.017)
    assert reason==f'empty_pickup_tool_robot:{PALM}:arm_left_3_link'
    assert 'proof' in f.events and f.events[-1][0]=='narrow'


def test_exact_sample_cache_and_cancel_precedence_unchanged():
    f=Fixture(); c=f.checker(); assert c.sample(f.q,.069) is None
    seen=list(f.events)
    assert c.sample(f.q,.069) is None and f.events==seen
    assert (c.checked_samples,c.cache_hits)==(1,1)
    f.node._cancel.set()
    assert c.sample(f.q,.069)=='empty_pickup_cancelled' and f.events==seen


def test_cached_rejection_and_original_narrow_exception_behavior_retained():
    f=Fixture(gap=-.1); c=f.checker()
    reason=c.sample(f.q,.069); before=list(f.events)
    c.last_rejection=None
    assert c.sample(f.q,.069)==reason and c.last_rejection==reason and f.events==before
    f=Fixture(gap=-.1); f.narrow_result=RuntimeError('original narrow failure')
    with pytest.raises(RuntimeError,match='original narrow failure'): f.checker().sample(f.q,.069)


def test_tool_and_robot_watertight_flags_forward_unchanged():
    f=Fixture(gap=-.1); f.watertight[PALM]=False; f.closed=set()
    assert f.checker().sample(f.q,.069)
    assert f.events[-1]==('narrow',dict(first_watertight=False,second_watertight=False))


def test_controller_edge_interior_rejection_is_not_replaced_by_clear_endpoints():
    f=Fixture(); c=f.checker(); end=f.q.copy(); end[2]=.4
    assert c.sample(f.q,.069) is None and c.sample(end,.069) is None
    assert c.edge(f.q,end,.069) is False
    assert 'tool_robot' in c.last_rejection
    assert 0.<c.last_rejected_q[2]<.4
