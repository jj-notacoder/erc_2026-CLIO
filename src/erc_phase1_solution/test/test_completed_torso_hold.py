"""Actual decorated sender/PLACE suffix with deterministic ROS action fixtures.

These tests prove software admission and command lifecycle, not physical hold,
external controller ownership, acceleration, or measured trial savings.
"""
import ast
import copy
import math
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution import completed_torso_hold as hold
from erc_phase1_solution import manipulation_node as runtime


def fixture(monkeypatch, enabled=True):
    sent, statuses, checks = [], [], []
    clock = [2_000_000_000]
    result = NS(done=lambda: True, result=lambda: NS(status=4))
    handle = NS(accepted=True, get_result_async=lambda: result)
    def send(goal):
        sent.append(copy.deepcopy(goal))
        return NS(result=lambda: handle)
    client = NS(wait_for_server=lambda **kw: True, send_goal_async=send)
    command = threading.RLock()
    n = NS(settled_place_torso_skip_enabled=enabled,
           _cancel=hold.TorsoHoldCancellation() if enabled else threading.Event(),
           _torso_hold_state=hold.TorsoHoldState(), _lock=threading.Lock(),
           _adaptive_command_guard=lambda: command, torso_client=client,
           arm_client=object(), head_client=object(), _goal_handles=[],
           _pending_retained_acceptances=set(), timeout=1.,
           get_clock=lambda: NS(now=lambda: NS(nanoseconds=clock[0])),
           joints={hold.JOINT:.35}, _joint_velocities={hold.JOINT:0.},
           _joint_stamps_ns={hold.JOINT:clock[0]},
           chain=NS(lower=np.array([0.]),upper=np.array([.35])),
           _wait_future=lambda f,t:f.result(), _cancel_goal_and_confirm=lambda *a:True,
           _publish_status=lambda event,**data:statuses.append((event,data)),
           _active_place_scene_reference=None, table_scene_required=True, bin_scene_required=True,
           _held_book_corners=np.arange(24,dtype=float).reshape(8,3)/1000,
           _contact_epoch=3,_contact_generation=5,_gripper_feedback_samples=[object()],
           _payload_hazard_reason=lambda **kw:None)
    monkeypatch.setattr(runtime,'time',NS(monotonic=lambda:0.,sleep=lambda _:None))
    monkeypatch.setattr(runtime,'_require_place_contact_clear',lambda n:checks.append('contact'))
    monkeypatch.setattr(runtime,'measured_scene_context',lambda n,r:checks.append('scene'))
    n._follow=MethodType(runtime.ManipulationNode._follow,n)
    n._move_torso=MethodType(runtime.ManipulationNode._move_torso,n)
    return n,sent,statuses,checks,clock,handle


def complete(n,clock,target=.35):
    assert n._move_torso(target,2.5)
    clock[0]+=20_000_000
    n.joints[hold.JOINT]=target
    n._joint_stamps_ns[hold.JOINT]=clock[0]
    n._active_place_scene_reference={'fresh':'fixture'}


@pytest.mark.parametrize('value',[True,False])
def test_boolean_parameter_only(value):
    assert hold.checked_torso_hold_enabled(value) is value


@pytest.mark.parametrize('value',[0,1,0.,1.,None,'false',[],math.nan])
def test_ambiguous_parameter_rejected(value):
    with pytest.raises(ValueError):hold.checked_torso_hold_enabled(value)


def test_default_keeps_normal_goal_and_never_creates_proof(monkeypatch):
    n,sent,statuses,_,clock,_=fixture(monkeypatch,False)
    complete(n,clock)
    assert n._move_torso(.35,2.2,allow_completed_hold=True)
    assert len(sent)==2 and n._torso_hold_state.completed is None
    point=sent[-1].trajectory.points[0]
    assert list(point.positions)==[.35]
    assert (point.time_from_start.sec,point.time_from_start.nanosec)==(2,200_000_000)
    assert not point.velocities and not point.accelerations
    assert not any(e=='torso_motion_skipped' for e,_ in statuses)


def test_actual_success_then_new_feedback_skips_only_explicit_place_site(monkeypatch):
    n,sent,statuses,checks,clock,_=fixture(monkeypatch)
    complete(n,clock)
    record=n._torso_hold_state.completed
    assert n._move_torso(.35,2.2,allow_completed_hold=True)
    assert len(sent)==1 and n._torso_hold_state.completed is record
    assert checks[-3:]==['contact','scene','contact']
    event,data=statuses[-1]
    assert event=='torso_motion_skipped' and data['producer_stamp_ns']>data['completed_action_ros_ns']
    assert data['target']==.35 and data['controller_command_sent'] is False
    assert n._move_torso(.35,2.2)  # Non-PLACE/recovery call always submits.
    assert len(sent)==2


@pytest.mark.parametrize('mode',['missing','same_stamp','old','future','moving','drift','nan','infinite_velocity','bool_stamp','different_target'])
def test_missing_or_bad_measurement_falls_back_same_normal_action(monkeypatch,mode):
    n,sent,statuses,_,clock,_=fixture(monkeypatch);complete(n,clock)
    target=.35
    if mode=='missing':n._torso_hold_state.completed=None
    elif mode=='same_stamp':n._joint_stamps_ns[hold.JOINT]=n._torso_hold_state.completed[3]
    elif mode=='old':clock[0]+=150_000_001
    elif mode=='future':n._joint_stamps_ns[hold.JOINT]=clock[0]+50_000_001
    elif mode=='moving':n._joint_velocities[hold.JOINT]=1.000001e-6
    elif mode=='drift':n.joints[hold.JOINT]-=1.000001e-6
    elif mode=='nan':n.joints[hold.JOINT]=math.nan
    elif mode=='infinite_velocity':n._joint_velocities[hold.JOINT]=math.inf
    elif mode=='bool_stamp':n._joint_stamps_ns[hold.JOINT]=True
    elif mode=='different_target':target=.34
    assert n._move_torso(target,2.2,allow_completed_hold=True)
    assert len(sent)==2 and list(sent[-1].trajectory.points[0].positions)==[target]
    assert not any(e=='torso_motion_skipped' for e,_ in statuses)


def test_tight_inclusive_position_velocity_and_age_boundaries(monkeypatch):
    n,sent,_,_,clock,_=fixture(monkeypatch);complete(n,clock,0.)
    n.joints[hold.JOINT]=1e-6;n._joint_velocities[hold.JOINT]=-1e-6
    clock[0]+=150_000_000
    assert n._move_torso(0.,2.2,allow_completed_hold=True) and len(sent)==1


@pytest.mark.parametrize('failure',['server','pre_send','refused','unknown_acceptance','aborted_result'])
def test_every_failed_or_uncertain_torso_attempt_invalidates_old_success(monkeypatch,failure):
    n,sent,_,_,clock,handle=fixture(monkeypatch);complete(n,clock)
    kwargs={}
    if failure=='server':n.torso_client.wait_for_server=lambda **kw:False
    elif failure=='pre_send':kwargs['pre_send_check']=lambda:(_ for _ in ()).throw(RuntimeError('pre-send failure'))
    elif failure=='refused':handle.accepted=False
    elif failure=='unknown_acceptance':n._wait_future=lambda f,t:None
    elif failure=='aborted_result':handle.get_result_async=lambda:NS(done=lambda:True,result=lambda:NS(status=6))
    try:assert n._follow(n.torso_client,[hold.JOINT],[.35],2.2,**kwargs) is False
    except RuntimeError:assert failure in ('server','pre_send')
    assert n._torso_hold_state.completed is None and not n._torso_hold_state.pending
    assert n._torso_hold_state.uncertain is (failure in ('refused','unknown_acceptance','aborted_result'))


def test_cancel_then_clear_cannot_revive_completed_hold(monkeypatch):
    n,sent,statuses,_,clock,_=fixture(monkeypatch);complete(n,clock)
    n._cancel.set();n._cancel.clear()
    assert n._move_torso(.35,2.2,allow_completed_hold=True) and len(sent)==2
    assert not any(e=='torso_motion_skipped' for e,_ in statuses)


def test_preexisting_cancel_invalidates_before_any_new_torso_action(monkeypatch):
    n,sent,_,_,clock,_=fixture(monkeypatch);complete(n,clock)
    n._cancel.set()
    assert n._move_torso(.35,2.2) is False and len(sent)==1
    assert n._torso_hold_state.completed is None


def test_out_of_order_torso_server_wait_cannot_publish_old_generation(monkeypatch):
    n,sent,_,_,clock,_=fixture(monkeypatch)
    entered,release=threading.Event(),threading.Event();errors=[]
    def wait(**kw):
        if threading.current_thread().name=='older':
            entered.set();assert release.wait(1.)
        return True
    n.torso_client.wait_for_server=wait
    def older():
        try:n._move_torso(.30,2.)
        except RuntimeError as e:errors.append(str(e))
    t=threading.Thread(target=older,name='older');t.start()
    try:
        assert entered.wait(1.)
        assert n._move_torso(.35,2.)
    finally:
        release.set();t.join(1.)
    assert not t.is_alive() and errors==['torso_hold_command_generation_changed']
    assert [list(g.trajectory.points[0].positions) for g in sent]==[[.35]]
    assert n._torso_hold_state.completed is None


def test_completion_from_older_call_cannot_overwrite_newer_generation(monkeypatch):
    n,sent,_,_,clock,_=fixture(monkeypatch)
    entered,release=threading.Event(),threading.Event();results=[]
    wait=n._wait_future
    def wait_future(f,timeout):
        if threading.current_thread().name=='older':
            entered.set();assert release.wait(1.)
        return wait(f,timeout)
    n._wait_future=wait_future
    t=threading.Thread(target=lambda:results.append(n._move_torso(.30,2.)),name='older');t.start()
    try:
        assert entered.wait(1.)
        assert n._move_torso(.35,2.)
    finally:
        release.set();t.join(1.)
    assert not t.is_alive() and results==[True] and len(sent)==2
    assert n._torso_hold_state.completed is None  # The overlap never proves a hold.


def test_records_serialized_target_after_caller_mutation_during_server_wait(monkeypatch):
    n,sent,_,_,clock,_=fixture(monkeypatch);positions=[.30]
    n.torso_client.wait_for_server=lambda **kw:(positions.__setitem__(0,.35) or True)
    assert n._follow(n.torso_client,[hold.JOINT],positions,2.)
    assert n._torso_hold_state.completed[2]==sent[-1].trajectory.points[0].positions[0]==.35


@pytest.mark.parametrize('pending',['active','retained','follow','uncertain'])
def test_no_skip_with_any_active_pending_or_unknown_action(monkeypatch,pending):
    n,_,statuses,_,clock,_=fixture(monkeypatch);complete(n,clock)
    if pending=='active':n._goal_handles.append(object())
    elif pending=='retained':n._pending_retained_acceptances.add(object())
    elif pending=='follow':n._torso_hold_state.pending.add(object())
    else:n._torso_hold_state.uncertain=True
    assert hold.try_completed_torso_hold(n,.35,require_contact=lambda:None,check_scene=lambda r:None) is None
    assert not any(e=='torso_motion_skipped' for e,_ in statuses)


@pytest.mark.parametrize('fault',['contact','scene','payload','latest_latch','cancel','attachment','reference','contact_epoch'])
def test_real_move_shortcut_preserves_hard_fault_veto_without_new_action(monkeypatch,fault):
    n,sent,_,_,clock,_=fixture(monkeypatch);complete(n,clock)
    if fault=='contact':monkeypatch.setattr(runtime,'_require_place_contact_clear',lambda n:(_ for _ in ()).throw(RuntimeError('contact')))
    elif fault=='payload':n._payload_hazard_reason=lambda **kw:'contact_lost'
    else:
        def scene(node,reference):
            if fault=='scene':raise RuntimeError('scene')
            if fault=='latest_latch':node._payload_hazard_latched='lost'
            elif fault=='cancel':node._cancel.set()
            elif fault=='attachment':node._held_book_corners[0,0]+=.01
            elif fault=='reference':node._active_place_scene_reference={}
            elif fault=='contact_epoch':node._contact_epoch+=1
        monkeypatch.setattr(runtime,'measured_scene_context',scene)
    if fault in ('contact','scene'):
        with pytest.raises(RuntimeError,match=fault):n._move_torso(.35,2.2,allow_completed_hold=True)
    else:assert n._move_torso(.35,2.2,allow_completed_hold=True) is False
    assert len(sent)==1


@pytest.mark.parametrize('change',['velocity','position','stamp','disordered_stamp','delivery','feedback','generation','cancel_then_clear'])
def test_between_guard_change_falls_back_instead_of_using_stale_snapshot(monkeypatch,change):
    n,sent,statuses,_,clock,_=fixture(monkeypatch);complete(n,clock)
    changed=[]
    def scene(node,reference):
        if changed:return
        changed.append(True)
        if change=='velocity':node._joint_velocities[hold.JOINT]=.01
        elif change=='position':node.joints[hold.JOINT]=.34
        elif change=='stamp':node._joint_stamps_ns[hold.JOINT]=clock[0]-150_000_001
        elif change=='disordered_stamp':node._joint_stamps_ns[hold.JOINT]-=1
        elif change=='delivery':node._contact_generation+=1
        elif change=='feedback':node._gripper_feedback_samples.append(object())
        elif change=='generation':node._torso_hold_state.generation+=1
        elif change=='cancel_then_clear':node._cancel.set();node._cancel.clear()
    monkeypatch.setattr(runtime,'measured_scene_context',scene)
    assert n._move_torso(.35,2.2,allow_completed_hold=True) and len(sent)==2
    assert not any(e=='torso_motion_skipped' for e,_ in statuses)


def test_actual_contiguous_place_suffix_preserves_both_probes_and_nominal_target(monkeypatch):
    n,sent,_,_,clock,_=fixture(monkeypatch);complete(n,clock)
    probes=[]
    n._fresh_retention_probe=lambda command,phase,**kw:(probes.append((command,phase,kw)) or True)
    n._best_effort=lambda function:function()
    n._recover_closed_place=lambda **kw:(_ for _ in ()).throw(AssertionError('unexpected recovery'))
    source=Path(runtime.__file__).read_text()
    tree=ast.parse(source)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
    place=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_place')
    def probe(node,phase):
        return any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)
                   and x.func.attr=='_fresh_retention_probe' and len(x.args)>1
                   and isinstance(x.args[1],ast.Constant) and x.args[1].value==phase
                   for x in ast.walk(node))
    start=next(i for i,x in enumerate(place.body) if probe(x,'pre_place_motion'))
    end=next(i for i,x in enumerate(place.body) if probe(x,'torso'))+1
    fn=ast.parse('def segment(self,place_torso_target,measured_torso_target,centered_target=None):\n return True').body[0]
    fn.body=copy.deepcopy(place.body[start:end])+[ast.Return(ast.Constant(True))]
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));scope={}
    exec(compile(module,'<actual contiguous PLACE torso suffix>','exec'),scope)
    assert scope['segment'](n,.35,None)
    assert probes==[('place','pre_place_motion',{}),('place','torso',{'leg':0})]
    assert len(sent)==1
