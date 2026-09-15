"""Prepared, unexecuted stationarity/lifecycle cases using real raw collectors.

No model loading or physics. Actual PLACE method wiring uses the existing
lightweight scene-flow fixture; the helper itself runs on a deterministic clock.
"""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import pytest
from erc_phase1_solution import preopen_stationary as gate
from erc_phase1_solution.release_sensor_adapter import record_raw_joints, record_raw_odom

ROOT = Path(__file__).resolve().parents[1]
NAMES = (*gate.IK_JOINTS, 'gripper_left_finger_joint')
GOAL = (.35, .1, .2, .3, .4, .5, .6, .7)
MASTER = .017


def raw(stamp=1_100_000_000):
    return dict(producer_stamp_ns=stamp, positions=dict(zip(NAMES, (*GOAL, MASTER))),
        velocities=dict.fromkeys(NAMES, 0.), sequence=1,
        odom=dict(stamp_ns=stamp, linear_speed=0., angular_speed=0.))


class Node:
    def __init__(self):
        self._lock = threading.Lock()
        self.command_lock = threading.RLock()
        self._cancel = threading.Event()
        self._goal_handles = []
        self._pending_retained_acceptances = []
        self._held_book_corners = [[x, y, z] for x in (-.1, .1)
                                  for y in (-.01, .01) for z in (-.15, .15)]
        self._contact_epoch = 4
        self._contact_generation = 0
        self._target_book_model = 'book_col_2_row_1_red'
        self._active_place_scene_reference = {'source': 'registered'}
        self._payload_monitor_enabled = True
        self._retention_probe_active = False
        self._gripper_open_confirmed = False
        self.table_scene_required = self.bin_scene_required = True
        self.gripper_open = .069
        self.grasp_contact_max_age = .15
        self.now = 1_000_000_000
        self.wall = 0.
        self.ticks = 0
        self.events = []
        self.contact_fault = None
        self.pinch = True
        self.feeds = True
        self.mutate = lambda sample: None
        self.on_tick = lambda: None
        self.on_context = lambda: None
        self.on_retention = lambda: None
        self.context_checks = 0

    def get_clock(self):
        return NS(now=lambda: NS(nanoseconds=self.now))

    def _adaptive_command_guard(self):
        return self.command_lock

    def require_sensor_available(self):
        # A broken implementation fails quickly rather than hanging a real Lock.
        assert self._lock.acquire(timeout=.05), 'sensor helper called under sensor lock'
        self._lock.release()

    def _payload_hazard_reason(self, *, max_age):
        self.require_sensor_available()
        assert max_age <= .15
        self.on_retention()
        return self.contact_fault

    def _pinch_sample(self, *, max_age):
        self.require_sensor_available()
        return self.pinch, MASTER, True, True, True

    def _publish_status(self, event, **fields):
        self.require_sensor_available()
        assert not self.command_lock._is_owned(), 'report held command lock'
        self.events.append((event, fields))

    def context(self, node, reference):
        assert node is self and reference is self._active_place_scene_reference
        self.require_sensor_available()
        self.context_checks += 1
        self.on_context()

    def feed(self, sample=None):
        sample = raw(self.now) if sample is None else sample
        self.mutate(sample)
        od = sample['odom']
        stamp = lambda ns: NS(sec=ns//1_000_000_000, nanosec=ns%1_000_000_000)
        record_raw_odom(self, NS(header=NS(stamp=stamp(od['stamp_ns'])),
            twist=NS(twist=NS(linear=NS(x=od['linear_speed'], y=0.),
                              angular=NS(z=od['angular_speed'])))))
        record_raw_joints(self, NS(header=NS(stamp=stamp(sample['producer_stamp_ns'])),
            name=list(NAMES), position=[sample['positions'].get(n, math.nan) for n in NAMES],
            velocity=[sample['velocities'].get(n, math.nan) for n in NAMES]))

    def sleep(self, seconds):
        assert not self.command_lock._is_owned(), 'wait held command lock'
        self.require_sensor_available()
        self.wall += .01
        self.now += 25_000_000
        self.ticks += 1
        self.on_tick()
        if self.feeds:
            self.feed()


@pytest.fixture
def node(monkeypatch):
    n = Node()
    monkeypatch.setattr(gate, 'time', NS(monotonic=lambda:n.wall, sleep=n.sleep))
    monkeypatch.setattr(gate, 'measured_scene_context', n.context)
    return n


def run(n):
    return gate.require_stationary_closed_pose(n, {'target_model':n._target_book_model,
        'trial_id':'trial', 'placement_attempt_id':'attempt'}, GOAL, MASTER)


def test_distinct_span_uses_actual_collectors_and_keeps_payload_monitor(node):
    result = run(node)
    assert result['producer_stamp_ns']-result['stationary_start_ns'] >= 100_000_000
    assert node.context_checks == 1 and node.ticks >= 5
    assert result['planned_master'] == MASTER
    assert node._payload_monitor_enabled and not node._retention_probe_active
    assert not node._delivery_measurement_active
    assert len(node.events) == 1 and node.events[0][1]['verified'] is True
    assert node.events[0][1]['book_stationarity_verified'] is False


@pytest.mark.parametrize('change,reason', [
    (lambda s:s['velocities'].__setitem__('arm_left_1_joint', .001001), 'arm_not'),
    (lambda s:s['velocities'].__setitem__('torso_lift_joint', .001001), 'arm_not'),
    (lambda s:s['positions'].__setitem__('arm_left_1_joint', GOAL[1]+.00201), 'arm_not'),
    (lambda s:s['positions'].__setitem__('torso_lift_joint', GOAL[0]+.00101), 'arm_not'),
    (lambda s:s['velocities'].__setitem__('gripper_left_finger_joint', .000101), 'master_not'),
    (lambda s:s['positions'].__setitem__('gripper_left_finger_joint', .069), 'master_not'),
    (lambda s:s['positions'].__setitem__('arm_left_2_joint', math.nan), 'joint_feedback_invalid'),
    (lambda s:s['velocities'].__setitem__('arm_left_2_joint', math.inf), 'joint_feedback_invalid'),
    (lambda s:s.__setitem__('producer_stamp_ns', 800_000_000), 'raw_joint'),
    (lambda s:s.__setitem__('producer_stamp_ns', 1_300_000_001), 'raw_joint'),
    (lambda s:s['odom'].__setitem__('stamp_ns', 800_000_000), 'base_not'),
    (lambda s:s['odom'].__setitem__('linear_speed', .00501), 'base_not'),
    (lambda s:s['odom'].__setitem__('angular_speed', .00801), 'base_not'),
    (lambda s:s['odom'].__setitem__('linear_speed', math.nan), 'base_not'),
])
def test_raw_boundaries_reject(change, reason):
    sample = raw(); change(sample)
    ok, why = gate.stationary_closed_sample(sample, GOAL, MASTER, 1_100_000_000)
    assert not ok and why.startswith(reason)


def test_future_data_only_negative_until_clock_catches_up():
    sample = raw(1_150_000_000)
    assert not gate.stationary_closed_sample(sample, GOAL, MASTER, 1_100_000_000)[0]
    assert gate.stationary_closed_sample(sample, GOAL, MASTER, 1_100_000_000, pending=True)[0]


def test_slow_simulation_catches_fixed_future_frame_without_short_wall_expiry(node, monkeypatch):
    node.get_clock=lambda:NS(ros_time_is_active=True, now=lambda:NS(nanoseconds=node.now))
    def slow_sleep(_):
        node.wall+=.01;node.now+=1_000_000;node.ticks+=1
        if node.ticks%25==0:node.feed(raw(node.now+100_000_000))
    monkeypatch.setattr(gate, 'time', NS(monotonic=lambda:node.wall, sleep=slow_sleep))
    result=run(node)
    diagnostics=node.events[-1][1]['timing_diagnostics']
    assert result['producer_stamp_ns']-result['stationary_start_ns']>=100_000_000
    assert node.wall>.5 and diagnostics['future_waits']>0
    assert diagnostics['future_wait_expired']==0
    assert not node._delivery_measurement_active


@pytest.mark.parametrize('case', ['no_data','duplicate_stamps','large_gaps','clock_stalled'])
def test_no_unproven_dwell_can_pass(node, case):
    if case == 'no_data': node.feeds=False
    elif case == 'duplicate_stamps': node.mutate=lambda s:s.__setitem__('producer_stamp_ns', 1_025_000_000)
    elif case == 'large_gaps':
        node.on_tick=lambda:setattr(node,'now',node.now+75_000_001)
    elif case == 'clock_stalled':
        node.on_tick=lambda:setattr(node,'now',1_000_000_000)
    with pytest.raises(gate.PreopenStationaryRejected, match='measurement_timeout'):
        run(node)
    assert node.wall <= 10.02 and not node._delivery_measurement_active
    assert node.events[-1][1]['verified'] is False


def test_between_poll_velocity_outlier_restarts_span(node):
    def feed_extra():
        if node.ticks == 4:
            sample=raw(node.now-1)
            sample['velocities']['arm_left_3_joint']=.2
            node.feed(sample)
    node.on_tick=feed_extra
    result=run(node)
    assert result['stationary_start_ns'] >= 1_100_000_000
    assert node.ticks >= 8


@pytest.mark.parametrize('case', ['cancel','active','pending','hazard','robot','raw_fault',
                                'epoch','attachment','target','monitor','probe','open','scene'])
def test_hard_faults_stop_before_success(node, case):
    def mutate():
        if node.ticks != 3:return
        if case=='cancel':node._cancel.set()
        elif case=='active':node._goal_handles.append(object())
        elif case=='pending':node._pending_retained_acceptances.append(object())
        elif case=='hazard':node._payload_hazard_latched='retention_hazard'
        elif case=='robot':node._target_robot_contact_latched=True
        elif case=='raw_fault':node._raw_contacts_first_failure={'reason':'wire failure'}
        elif case=='epoch':node._contact_epoch+=1
        elif case=='attachment':node._held_book_corners[0][0]+=.001
        elif case=='target':node._target_book_model='other'
        elif case=='monitor':node._payload_monitor_enabled=False
        elif case=='probe':node._retention_probe_active=True
        elif case=='open':node._gripper_open_confirmed=True
        elif case=='scene':node._active_place_scene_reference={}
    node.on_tick=mutate
    with pytest.raises(gate.PreopenStationaryRejected):run(node)
    assert not any(fields['verified'] for _,fields in node.events)
    assert not node._delivery_measurement_active


@pytest.mark.parametrize('fault', ['contact','pinch','context','cancel_in_context','outlier_in_context','odom_in_context'])
def test_final_admission_and_sensor_races(node, fault):
    if fault=='contact':node.contact_fault='contact_lost'
    elif fault=='pinch':node.pinch=False
    elif fault=='context':
        node.on_context=lambda:(_ for _ in ()).throw(RuntimeError('scene moved'))
    elif fault=='cancel_in_context':node.on_context=node._cancel.set
    elif fault=='outlier_in_context':
        def disturb():
            sample=raw(node.now)
            sample['velocities']['arm_left_1_joint']=.2
            node.feed(sample)
        node.on_context=disturb
    else:
        def disturb_odom():
            with node._lock:
                node._delivery_raw_odom['linear_speed']=.1
        node.on_context=disturb_odom
    with pytest.raises(RuntimeError):run(node)
    assert not any(fields['verified'] for _,fields in node.events)
    assert not node._delivery_measurement_active


def test_contact_delivery_during_final_helpers_retries_without_mixed_admission(node):
    def update_once():
        if node.context_checks==1:node._contact_generation+=1
    node.on_context=update_once
    result=run(node)
    assert node.context_checks==2 and result['contact_generation']==1


def test_clock_reversal_is_hard_failure(node):
    node.on_tick=lambda:setattr(node,'now',900_000_000) if node.ticks==3 else None
    with pytest.raises(gate.PreopenStationaryRejected, match='clock_reversed'):run(node)


def test_joint_stamp_reversal_is_hard_failure(node):
    def reverse(sample):
        if node.ticks==3:sample['producer_stamp_ns']=1_040_000_000
    node.mutate=reverse
    with pytest.raises(gate.PreopenStationaryRejected,match='joint_stamp_reversed'):run(node)


def test_raw_ring_overflow_cannot_certify_missing_span(node):
    def overflow():
        if node.ticks==4:
            for _ in range(140):node.feed(raw(node.now))
    node.on_tick=overflow
    result=run(node)
    assert result['stationary_start_ns'] >= 1_100_000_000
    assert node.ticks >= 8


@pytest.mark.parametrize('case',['epoch','attachment','pending','hazard'])
def test_final_hard_interlock_update_cannot_pass(node,case):
    def change():
        if case=='epoch':node._contact_epoch+=1
        elif case=='attachment':node._held_book_corners[0][2]+=.01
        elif case=='pending':node._pending_retained_acceptances.append(object())
        else:node._held_grip_sensor_fault='invalid feedback'
    node.on_context=change
    with pytest.raises(gate.PreopenStationaryRejected):run(node)
    assert not any(fields['verified'] for _,fields in node.events)


def test_existing_measurement_is_not_stolen_or_deactivated(node):
    node._delivery_measurement_active=True
    with pytest.raises(gate.PreopenStationaryRejected, match='already_active'):run(node)
    assert node._delivery_measurement_active


@pytest.mark.parametrize('master', [None, math.nan, math.inf, -.001, .069])
def test_invalid_or_open_reference_never_enables_sampler(node, master):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        gate.require_stationary_closed_pose(node, None, GOAL, master)
    assert not getattr(node,'_delivery_measurement_active',False)


def test_status_failure_does_not_authorize_rejected_gate(node):
    node._publish_status=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('publisher'))
    node.contact_fault='contact_lost'
    with pytest.raises(gate.PreopenStationaryRejected,match='contact_lost'):run(node)


def test_whole_node_restores_exact_parent_and_default_false():
    from candidate_composition_support import restore_preopen_speed_source
    text=restore_preopen_speed_source((ROOT/'erc_phase1_solution/manipulation_node.py').read_text())
    inverse=json.loads((ROOT/'test/fixtures/preopen_stationary_inverse.json').read_text())
    assert "'preopen_stationary_enabled': False" in text
    for insertion in inverse['insertions']:
        assert text.count(insertion)==1
        text=text.replace(insertion,'',1)
    assert hashlib.sha256(text.encode()).hexdigest()==inverse['parent_node_sha256']


@pytest.mark.parametrize('enabled,outcome', [(False,'unused'),(True,'ok'),(True,'reject'),(True,'error')])
def test_actual_place_gate_orders_after_route_before_open_and_return(monkeypatch, enabled, outcome):
    from test_scene_checked_place_flow import PlaceWiring
    fixture=PlaceWiring();fixture.setUp()
    node=fixture.node
    node.preopen_stationary_enabled=enabled
    def require(n,identity,goal,master):
        assert n is node
        fixture.calls.append(('preopen',(list(goal),master)))
        if outcome=='reject':raise gate.PreopenStationaryRejected('unconfirmed')
        if outcome=='error':raise RuntimeError('sensor failed')
    monkeypatch.setattr(gate,'require_stationary_closed_pose',require)
    # The complete original method is AST-compiled, retaining its relative import.
    fixture.namespace['__package__']='erc_phase1_solution'
    if outcome in ('reject','error'):
        with pytest.raises(RuntimeError):fixture.namespace['_place'](node)
        assert not any(kind in ('open','return') for kind,_ in fixture.calls)
    else:
        assert fixture.namespace['_place'](node)
        kinds=[kind for kind,_ in fixture.calls]
        if enabled:
            assert kinds.index('execute') < kinds.index('preopen') < kinds.index('open') < kinds.index('return')
        else:assert 'preopen' not in kinds
