"""Actual ROS methods/callbacks: open -> stock close -> PICK attach -> PLACE.

Only the unchanged expensive PICK planning prefix and PLACE planning prefix are
excluded. Their exact contiguous execution suffixes are pinned, not rewritten.
No simulator, model construction, action servers or physical retention claim.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace as NS

import numpy as np
import pytest
from erc_phase1_solution import manipulation_node as node_module
from erc_phase1_solution import preopen_stationary as gate
from nav_msgs.msg import Odometry
from test_stock_gripper_close import close_fixture

ROOT = Path(__file__).resolve().parents[1]
RECORD = json.loads((ROOT/'test/fixtures/stock_close_lifecycle_inverse.json').read_text())
GOAL = np.asarray([.35, 0., 0., 0., 0., 0., 0., 0.])
MODEL = 'book_col_2_row_1_red'
MASTER = .017


def execution_suffix(name):
    text = (ROOT/'erc_phase1_solution/manipulation_node.py').read_text()
    owner = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)
                 and n.name == 'ManipulationNode')
    function = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = function.body
    if name == '_pick':
        start = next(i for i,n in enumerate(body) if isinstance(n, ast.Try)
                     and 'close_result = self._close_for_grasp()' in ast.get_source_segment(text,n))
        end = next(i for i,n in enumerate(body) if isinstance(n, ast.AnnAssign)
                   and isinstance(n.target, ast.Name) and n.target.id == 'carried_legs')
        body = body[start:end]
        expected = RECORD['retained_pick_slice_sha256']
    else:
        start = next(i for i,n in enumerate(body) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "not self._fresh_retention_probe('place', 'pre_place_motion')")
        body = body[start:]
        # Bind the actual suffix, including guarded torso hold, loaded-only timing cap, raised normal finish, and scoped clearance timing.
        # Historical inverse fixture bytes remain pinned by structural tests.
        expected = 'fb41148d07c9af75103598c6de35497209491e8213cd1b058935e52b483420a9'
    exact = '\n'.join(text.splitlines()[body[0].lineno-1:body[-1].end_lineno])+'\n'
    assert hashlib.sha256(exact.encode()).hexdigest() == expected
    # Retain every statement in the original contiguous suffix. Extra return
    # only exposes PICK's resulting verified flag to the regression caller.
    result = copy.deepcopy(function)
    result.body = copy.deepcopy(body)
    if name == '_place':
        initialization = function.body[0]
        assert isinstance(initialization, ast.Assign)
        assert [target.id for target in initialization.targets] == ['release_only_request', 'release_only_endpoint']
        assert isinstance(initialization.value, ast.Constant) and initialization.value.value is None
        result.body.insert(0, copy.deepcopy(initialization))
    if name == '_pick':
        result.body.append(ast.Return(value=ast.Name(id='grasp_verified',ctx=ast.Load())))
    scope = dict(vars(node_module))
    scope.update(interval_enabled=False, attached_corners=np.asarray([
        [x,y,z] for x in (-.1,.1) for y in (-.01,.01) for z in (-.15,.15)]),
        planned_post_retreat_plan=None, solutions=[GOAL.copy(),GOAL.copy()],
        transition_waypoints=[], measured_torso_target=None, place_torso_target=.35,
        carried_transition_waypoints=[], torso_ready=GOAL.copy(),
        unloaded_home_waypoints=[], direct_empty_home=None,
        place_scene_diagnostics={'measured_master':MASTER}, correlation=None)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')],level=0),result],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'pinned actual execution suffix','exec'),scope)
    return scope[name], scope


@pytest.fixture
def lifecycle(monkeypatch):
    unused,n,now,wall,sent,events,sensors = close_fixture(monkeypatch)
    n._stock_close_controller = None
    n._adaptive_close_active = False
    n._adaptive_motion_started = False
    n.adaptive_close_step = .001  # Select the actual production dispatcher.
    n.stock_gripper_close_target_position = MASTER
    n.gripper_open = .069
    n.gripper_settle = .08
    n.timeout = 2.
    n.grasp_contact_max_age = .15
    n.grasp_max_position = .05
    n._payload_monitor_enabled = False
    n._retention_probe_active = False
    n._gripper_open_confirmed = False
    n._active_place_scene_reference = None
    n._held_book_corners = None
    n.preopen_stationary_enabled = True
    n.delivery_evidence_enabled = True  # Actual selected R55 callback admission.
    n.bin_scene_required = n.table_scene_required = True
    n._test_send_failure = None
    n._test_wait_failure = False
    n._test_close_publications = []
    n._test_returns = []
    n._test_recoveries = []
    n._best_effort = lambda function:function()
    n._move_torso = lambda *a,**kw:True
    n._execute_retained_arm_legs = lambda legs,*a,**kw:(True,len(legs),False)
    n._return_from_bin = lambda *a,**kw:n._test_returns.append(True) or True
    n._recover_ambiguous_grasp = lambda *a:n._test_recoveries.append('pick') or False
    n._recover_closed_place = lambda **kw:n._test_recoveries.append(kw) or False

    def publish(message):
        # Inspect the actual ROS trajectory at the public publication boundary.
        assert not n._lock.locked(), 'publication must release sensor lock'
        q = message.points[0].positions[0]
        point = message.points[0]
        stock_command = (n._stock_close_controller is not None
            and point.time_from_start.sec == 1 and point.time_from_start.nanosec == 0)
        if stock_command:
            assert n._adaptive_command_guard()._is_owned()
            n._test_close_publications.append((q,n._gripper_open_confirmed))
            if n._test_send_failure:
                raise RuntimeError(n._test_send_failure)
        sent.append(message)
        n.joints['gripper_left_finger_joint'] = q
    n.gripper_pub = NS(publish=publish)

    def fresh_sensors():
        message = Odometry()
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(now.nanoseconds,10**9)
        message.header.frame_id = 'odom'
        message.child_frame_id = 'base_footprint'
        message.pose.pose.position.x = 2.
        message.pose.pose.position.y = -.1
        message.pose.pose.orientation.w = 1.
        n._on_staging_odom(message)  # Real callback, with selected collection flag.
        sensors()  # Unchanged actual _on_joint_state + both _on_contacts.
    wall.hook = fresh_sensors
    original_wait = n._wait_adaptive_gripper_duration
    def wait(duration):
        if n._test_wait_failure:return False
        return original_wait(duration)
    n._wait_adaptive_gripper_duration = wait
    def context(node, reference):
        assert node is n and reference is n._active_place_scene_reference
        assert n._lock.acquire(timeout=.05), 'context called under plain sensor lock'
        n._lock.release()
    monkeypatch.setattr(gate,'measured_scene_context',context)
    monkeypatch.setattr(node_module,'measured_scene_context',context)
    pick, pick_scope = execution_suffix('_pick')
    place, place_scope = execution_suffix('_place')
    def prepare_place(correlated=True):
        n._active_place_scene_reference = {'source':'registered test context'}
        place_scope['correlation'] = ({'target_model':n._target_book_model,
            'trial_id':'trial','placement_attempt_id':'attempt'} if correlated else None)
        place_scope['observe_measured_open_pose'] = lambda *a:{'verified':True}
    return NS(n=n,now=now,wall=wall,sent=sent,events=events,
        pick=lambda:pick(n), place=lambda:place(n), prepare_place=prepare_place,
        sensors=fresh_sensors)


@pytest.mark.parametrize('correlated',[True,False])
def test_real_open_stock_pick_retention_preopen_and_both_place_branches(lifecycle,correlated):
    f=lifecycle;n=f.n
    assert n._open_gripper()
    assert n._gripper_open_confirmed and not n._payload_monitor_enabled
    assert f.pick()
    assert n._test_close_publications == [(MASTER,False)]
    assert not n._gripper_open_confirmed
    assert n._payload_monitor_enabled and not n._retention_probe_active
    assert n._held_book_corners is not None and n._stock_close_controller is None
    f.prepare_place(correlated)
    assert f.place()
    admitted = [(i,v) for i,(event,v) in enumerate(f.events)
                if event=='placement_preopen_stationary']
    assert len(admitted)==1 and admitted[0][1]['verified']
    assert n._delivery_sample_sequence >= 51  # Actual 2ms callbacks span 100ms.
    assert n._delivery_raw_samples and n._delivery_raw_odom is not None
    assert admitted[0][1]['producer_stamp_ns']-admitted[0][1]['stationary_start_ns']>=100_000_000
    assert n._test_returns == [True]
    assert n._gripper_open_confirmed and n._held_book_corners is None
    assert not n._payload_monitor_enabled and not n._retention_probe_active
    assert [m.points[0].positions[0] for m in f.sent] == [.069,.069,MASTER,.069,.069]


def test_exact_r55_publisher_reproduces_stale_open_failure(lifecycle):
    f=lifecycle;n=f.n
    scope=dict(vars(node_module))
    exec('from __future__ import annotations\n'+RECORD['original_publisher'],scope)
    n._publish_adaptive_gripper_position=MethodType(scope['_publish_adaptive_gripper_position'],n)
    assert n._open_gripper() and f.pick()
    assert n._test_close_publications==[(MASTER,True)]
    assert n._payload_monitor_enabled and not n._retention_probe_active
    f.prepare_place()
    with pytest.raises(gate.PreopenStationaryRejected,match='gripper_open_still_confirmed'):f.place()
    assert len(f.sent)==3 and not n._test_returns
    assert f.events[-1][1]['verified'] is False


@pytest.mark.parametrize('failure',['publish','wait','contact'])
def test_failed_stock_close_never_installs_retained_state(lifecycle,failure):
    f=lifecycle;n=f.n
    assert n._open_gripper()
    if failure=='publish':n._test_send_failure='publication failed'
    elif failure=='wait':n._test_wait_failure=True
    else:n._adaptive_pressure_evidence=lambda **kw:(_ for _ in ()).throw(RuntimeError('contact failed'))
    assert not f.pick()
    assert not n._gripper_open_confirmed and n._held_book_corners is None
    assert not n._payload_monitor_enabled and not n._adaptive_close_active
    assert n._stock_close_controller is None and n._test_recoveries==['pick']
    assert not n._test_returns


@pytest.mark.parametrize('failure',['cancel','feedback'])
def test_prepublication_rejection_does_not_erase_known_open_proof(lifecycle,failure):
    f=lifecycle;n=f.n
    assert n._open_gripper()
    if failure=='cancel':n._cancel.set()
    else:n._adaptive_motion_feedback=lambda:(None,'invalid_feedback')
    assert not f.pick()
    assert n._gripper_open_confirmed and not n._test_close_publications
    assert n._held_book_corners is None and not n._payload_monitor_enabled


@pytest.mark.parametrize('correlated',[True,False])
@pytest.mark.parametrize('failure',['contact','cancel','velocity','monitor','probe','open'])
def test_postclose_gate_fault_never_opens_or_returns(lifecycle,correlated,failure):
    f=lifecycle;n=f.n
    assert n._open_gripper() and f.pick()
    f.prepare_place(correlated)
    original=n._execute_retained_arm_legs
    def endpoint(*a,**kw):
        result=original(*a,**kw)
        if failure=='contact':n._payload_hazard_latched='contact_lost'
        elif failure=='cancel':n._cancel.set()
        elif failure=='velocity':
            original_joint=n._on_joint_state
            def moving_joint(message):
                message.velocity[message.name.index('arm_left_1_joint')]=.0011
                original_joint(message)
            n._on_joint_state=moving_joint
        elif failure=='monitor':n._payload_monitor_enabled=False
        elif failure=='probe':n._retention_probe_active=True
        else:n._gripper_open_confirmed=True
        return result
    n._execute_retained_arm_legs=endpoint
    with pytest.raises(RuntimeError):f.place()
    assert len(f.sent)==3 and not n._test_returns and n._held_book_corners is not None
    event=next(v for e,v in reversed(f.events) if e=='placement_preopen_stationary')
    assert not event['verified']
    if failure=='velocity':
        assert event['reason']=='measurement_timeout:arm_not_at_stationary_checked_endpoint'
    if failure in ('monitor','probe','open'):
        assert event['reason']=={'monitor':'payload_monitor_disabled',
            'probe':'retention_probe_active','open':'gripper_open_still_confirmed'}[failure]


@pytest.mark.parametrize('damage',['missing','duplicate','unrelated'])
def test_strict_node_inverse_rejects_extra_or_missing_changes(damage):
    from candidate_composition_support import restore_stock_close_lifecycle_source
    text=(ROOT/'erc_phase1_solution/manipulation_node.py').read_text()
    item=RECORD['node_inverse']
    if damage=='missing':text=text.replace(item['after'],item['before'],1)
    elif damage=='duplicate':text+=item['after']
    else:text+='\n# undeclared edit\n'
    with pytest.raises(AssertionError):restore_stock_close_lifecycle_source(text)


def test_gate_only_splits_the_same_three_hard_interlocks():
    from candidate_composition_support import restore_preopen_timing_source
    text=restore_preopen_timing_source((ROOT/'erc_phase1_solution/preopen_stationary.py').read_text())
    item=RECORD['gate_inverse']
    assert text.count(item['after'])==1
    text=text.replace(item['after'],item['before'],1)
    assert hashlib.sha256(text.encode()).hexdigest()==item['parent_sha256']


def test_disabled_raw_collection_fails_closed_without_opening(lifecycle):
    f=lifecycle;n=f.n
    assert n._open_gripper() and f.pick()
    n.delivery_evidence_enabled=False
    f.prepare_place()
    with pytest.raises(gate.PreopenStationaryRejected,
            match='measurement_timeout:measurement_timeout'):f.place()
    assert n._delivery_sample_sequence==0 and not n._delivery_raw_samples
    assert n._delivery_raw_odom is None and not n._delivery_measurement_active
    assert len(f.sent)==3 and not n._test_returns and n._held_book_corners is not None
