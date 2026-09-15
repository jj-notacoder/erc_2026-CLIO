"""Fine-pressure controller regressions with real ROS messages and fake time.

No nodes, actuators or simulated world are started. Pressure success is not
treated as loaded retention; the ordinary pick path owns that later proof.
"""
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
from erc_phase1_solution import fine_gripper_close as fine
from erc_phase1_solution import manipulation_node as m
from erc_phase1_solution.adaptive_grasp import ForceSample, GripperFeedback
from test_adaptive_gripper_interrupt import motion_node
from test_contact_force_timing import BOOK, LEFT, RIGHT, force_message


class Clock:
    def __init__(self, now):
        self.now, self.wall, self.on_sleep = now, 0., None
        self.advancing = True

    def monotonic(self):
        return self.wall

    def sleep(self, seconds):
        self.wall += seconds
        if self.advancing:
            self.now.nanoseconds += round(seconds*1e9)
        if self.on_sleep:
            self.on_sleep()


def fixture(monkeypatch, *, model=None, colour='red', position=.01825, limits=None):
    node, now, messages, statuses = motion_node()
    clock = Clock(now)
    monkeypatch.setattr(fine, 'time', clock)
    node.target_colour, node._target_book_model = colour, model
    node.gripper_open, node._gripper_open_confirmed = .069, True
    node.adaptive_start_tolerance = .0015
    node.adaptive_unilateral_travel_limit, node.adaptive_endpoint_tolerance = .004, .0005
    node.adaptive_contact_force_minimum, node.adaptive_contact_samples = .05, 3
    node.adaptive_contact_max_age, node.adaptive_contact_max_gap = .15, .075
    node.adaptive_contact_min_span, node.adaptive_contact_max_skew = .05, .05
    node.adaptive_velocity_tolerance = .005
    node.grasp_min_position, node.grasp_min_margin, node.grasp_max_position = .0155, .0005, .055
    node.gripper_transport_lock = .015
    node.count_publishers = lambda topic: 1
    node.gripper_pub.get_subscription_count = lambda: 1
    if limits is not None:
        node.adaptive_contact_force_maximum = limits.maximum_force
    set_feedback(node, now, position)
    controller = fine.FineGripperClose(node, limits)
    node._fine_gripper_controller = controller
    controller._fine_phase = 'fine'
    return controller, node, now, clock, messages, statuses


def set_feedback(node, now, position, velocity=0., stamp=None):
    node._gripper_feedback_samples.append(GripperFeedback(
        now.nanoseconds if stamp is None else stamp, position, velocity, .1))
    node.joints['gripper_left_finger_joint'] = position


def micro_fixture(monkeypatch, *, limits=None):
    c, node, now, clock, messages, statuses = fixture(
        monkeypatch, model='book_col_2_row_4_red', limits=limits)
    c._fine_stop = 'bilateral_contact_observed'
    c._fine_command = .01825
    c._fine_side_last = {'right': (now.nanoseconds, .3)}
    c._fine_first_contact = .0183
    c._fine_micro_reference = .01825
    c._fine_micro_stationary_start_ns = now.nanoseconds-100_000_000
    return c, node, now, clock, messages, statuses


@pytest.mark.parametrize('overrides', [
    {'micro_preload_steps': 51}, {'micro_preload_steps': True},
    {'micro_preload_step': .000000501}, {'micro_preload_limit': .00007501},
    {'micro_position_error': .000000051}, {'minimum_force': 8.},
    {'minimum_force': .049}, {'micro_preload_steps': 20, 'micro_preload_limit': .000001},
    {'stationary_velocity': .0000021}, {'total_wall_seconds': 1500.1},
    {'maximum_force': 30.0001}, {'floor': float('nan')},
])
def test_limits_cannot_expand_the_bounded_profile(overrides):
    with pytest.raises(ValueError):
        fine.FineGripLimits(**overrides)


def test_explicit_bounded_candidate_does_not_change_default_force_policy():
    default = fine.FineGripLimits()
    candidate = replace(default, maximum_force=20., minimum_force=6.)
    assert default.maximum_force == 8. and default.minimum_force == 1.
    assert candidate.maximum_force == 20. and candidate.minimum_force == 6.
    assert candidate.micro_preload_limit == default.micro_preload_limit == 2e-6
    assert candidate.micro_position_error == default.micro_position_error == 25e-9


@pytest.mark.parametrize('maximum,force,fault', [
    (8., 8.1, True), (20., 10.3476, False), (20., 20., False),
    (20., 20.0001, True), (20., 21., True),
    (30., 20.3735, False), (30., 30., False), (30., 30.0001, True),
])
def test_configured_force_cap_applies_to_real_callbacks_and_latched_hold(
        monkeypatch, maximum, force, fault):
    limits = fine.FineGripLimits(maximum_force=maximum)
    c,node,now,_,messages,_ = fixture(monkeypatch, limits=limits)
    node._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,3.),(RIGHT,BOOK,3.)))
    assert c._fine_stop == 'bilateral_contact_observed' and c._fine_fault is None
    node._on_contacts(force_message(now.nanoseconds+2_000_000,(LEFT,BOOK,force)))
    assert (c._fine_fault == 'force_overload') is fault
    assert (node._adaptive_overload_reason() == 'force_overload') is fault
    assert (node._adaptive_motion_halt_reason == 'force_overload') is fault
    if fault:
        before = len(messages)
        c._run_micro_preload()
        with c._fine_lock:
            assert not c._publish_once_locked(.017, .2, 'close')
        assert len(messages) == before


@pytest.mark.parametrize('model', ['book_col_2_row_4_blue', 'red',
                                 'book_col_2_row_4_red::collision'])
def test_invalid_inherited_identity_rejected_before_any_command(monkeypatch, model):
    with pytest.raises(ValueError):
        fixture(monkeypatch, model=model)


@pytest.mark.parametrize('colour,model', [('blue','book_col_1_row_5_blue'),
                                        ('green','book_col_5_row_3_green')])
def test_normal_concrete_colour_identity_has_no_fixed_seed_or_book(monkeypatch, colour, model):
    c, node, now, _, messages, _ = fixture(monkeypatch, colour=colour)
    book = model+'::book_base_link::collision'
    node._on_contacts(force_message(now.nanoseconds, (LEFT, book, 2.), (RIGHT, book, 2.)))
    assert c.expected_model == node._target_book_model == model
    assert c._fine_stop == 'bilateral_contact_observed'
    assert c._fine_fault is None
    assert len(messages) == 1 and list(messages[0].points[0].positions) == [.01825]


@pytest.mark.parametrize('second', ['book_col_2_row_4_red::book_base_link::collision',
                                   'book_col_3_row_2_blue::book_base_link::collision',
                                   'anonymous_book::collision'])
def test_second_different_or_anonymous_model_cannot_form_bilateral_contact(monkeypatch, second):
    c, node, now, _, messages, _ = fixture(monkeypatch)
    node._on_contacts(force_message(now.nanoseconds, (LEFT, BOOK, .3)))
    node._on_contacts(force_message(now.nanoseconds, (RIGHT, second, .3)))
    assert c._fine_fault == 'unexpected_or_anonymous_hand_contact'
    assert not any(list(msg.points[0].positions) == [.017] for msg in messages)
    with c._fine_lock:
        assert not c._publish_once_locked(.017, .2, 'close')


def test_first_bilateral_hold_does_not_set_fault_but_later_overload_wins(monkeypatch):
    c, node, now, _, messages, _ = fixture(monkeypatch)
    node._on_contacts(force_message(now.nanoseconds, (LEFT, BOOK, 2.), (RIGHT, BOOK, 2.)))
    assert c._fine_fault is None and node._adaptive_motion_halt_reason is None
    node._on_contacts(force_message(now.nanoseconds+2_000_000, (LEFT, BOOK, 8.1)))
    assert c._fine_fault == 'force_overload'
    assert node._adaptive_motion_halt_reason == 'force_overload'
    before = len(messages)
    with c._fine_lock:
        assert not c._publish_once_locked(.017, .2, 'close')
    assert len(messages) == before and not node._cancel.is_set()


def test_already_latched_callback_fault_blocks_queued_publish(monkeypatch):
    c, node, _, _, messages, _ = fixture(monkeypatch)
    node._adaptive_overload_latched = 'force_overload'
    with c._fine_lock:
        assert not c._publish_once_locked(.017, .2, 'close')
    assert len(messages) == 1 and list(messages[0].points[0].positions) == [.01825]
    assert c._fine_fault == 'force_overload'


def test_callback_waiting_for_finalizer_lock_cannot_mutate_retired_controller(monkeypatch):
    c,node,now,_,messages,_=fixture(monkeypatch)
    entered=threading.Event()
    def callback():
        entered.set()
        c.observe_contacts(force_message(now.nanoseconds,(LEFT,BOOK,9.)))
    with c._fine_lock:
        thread=threading.Thread(target=callback)
        thread.start()
        assert entered.wait(1.)
        c.active=False
        node._fine_gripper_controller=None
        node._adaptive_close_active=False
    thread.join(timeout=1.)
    assert not thread.is_alive()
    assert c.expected_model is None and c._fine_side_last=={}
    assert c._fine_stop is None and c._fine_fault is None
    assert not messages


def test_first_contact_during_preclose_is_a_permanent_fault(monkeypatch):
    c, node, now, _, _, _ = fixture(monkeypatch)
    c._fine_phase = 'preclose'
    node._on_contacts(force_message(now.nanoseconds, (RIGHT, BOOK, .3)))
    assert c._fine_fault == 'contact_during_coarse_seek'


def test_fine_stationarity_tolerates_measured_jitter_only_after_motion(monkeypatch):
    c, node, now, clock, _, statuses = fixture(monkeypatch)
    clock.on_sleep = lambda: set_feedback(node, now, .01825+3e-9, 1.34e-6)
    assert c._step(.01825, 'fine')
    endpoint = next(f for e,f in statuses if f.get('stage')=='stationary_endpoint')
    assert clock.wall >= .4
    assert endpoint['stationary_start_ros_ns'] >= 10_200_000_000


def test_micro_target_cannot_settle_using_the_larger_ordinary_tolerance(monkeypatch):
    c, node, now, clock, _, statuses = fixture(monkeypatch)
    clock.on_sleep = lambda: set_feedback(node, now, .01825)
    assert not c._wait_motion_and_stationary(.01825-1e-7, micro=True)
    assert c._fine_fault == 'step_wall_timeout'
    assert not any(f.get('stage')=='stationary_endpoint' for _,f in statuses)


@pytest.mark.parametrize('case', ['cancelled','clock_stalled','wall_timeout','missing_feedback'])
def test_faults_bound_waits_and_never_continue_inward(monkeypatch, case):
    c, node, now, clock, messages, _ = fixture(monkeypatch)
    clock.on_sleep = lambda: set_feedback(node, now, .01825)
    if case=='cancelled': node._cancel.set()
    elif case=='clock_stalled': clock.advancing=False
    elif case=='wall_timeout': clock.wall=300.
    else:
        node._gripper_feedback_samples.clear()
        clock.on_sleep=None
    assert not c._step(.017, 'fine')
    assert c._fine_fault
    assert clock.wall <= (300.01 if case=='wall_timeout' else 1.01)
    assert not messages or list(messages[-1].points[0].positions)==[.01825]


def test_effective_unilateral_guard_keeps_existing_endpoint_margin(monkeypatch):
    c, node, now, _, _, _ = fixture(monkeypatch)
    c._fine_first_contact=.022
    c._fine_side_last={'right':(now.nanoseconds,.3)}
    set_feedback(node,now,.0185-1e-12)
    assert c._feedback_error()=='unilateral_contact_travel_limit'


def pressure_fixture(monkeypatch, *, limits=None):
    c, node, now, clock, messages, statuses = micro_fixture(monkeypatch, limits=limits)
    c._fine_stop=None
    c._fine_micro_active=True
    return c,node,now,clock,messages,statuses


@pytest.mark.parametrize('span,verified', [(20,False),(62,True)])
def test_one_newton_pressure_needs_fifty_milliseconds_of_producer_history(monkeypatch,span,verified):
    c,node,now,_,_,statuses=pressure_fixture(monkeypatch)
    samples=tuple(ForceSample(now.nanoseconds-offset*1_000_000,
                             1.2 if offset<=span else .1) for offset in range(62,-1,-2))
    node._adaptive_force_histories=lambda:(c.expected_model,samples,samples,None)
    legacy,pressure=c._micro_pressure_snapshot(1,.01825)
    assert legacy.verified and pressure.verified is verified
    record=next(f for _,f in statuses if f.get('stage')=='pressure_measurement')
    assert record['pressure_windows']['left']['span_seconds']==pytest.approx(span/1000.)
    assert record['retention_verified'] is False


@pytest.mark.parametrize('left_force,right_force,span,verified', [
    (6.1, 6.2, 62, True), (6., 6., 62, True),
    (5.999, 6.2, 62, False), (6.1, 5.999, 62, False),
    (6.1, 6.2, 20, False), (21., 6.2, 62, False),
])
def test_six_newton_candidate_still_requires_both_producer_windows_and_force_cap(
        monkeypatch, left_force, right_force, span, verified):
    limits = fine.FineGripLimits(maximum_force=20., minimum_force=6.)
    c,node,now,_,messages,statuses = pressure_fixture(monkeypatch, limits=limits)
    def samples(force):
        return tuple(ForceSample(now.nanoseconds-offset*1_000_000,
                                 force if offset<=span else .1)
                     for offset in range(62,-1,-2))
    node._adaptive_force_histories = lambda:(c.expected_model,samples(left_force),samples(right_force),None)
    result = c._micro_pressure_snapshot(1,.01825)
    if max(left_force, right_force) > limits.maximum_force:
        # Hard raw overload must stop before a soft settling path can discard
        # its history, rather than returning ordinary deficient pressure.
        assert result is None and verified is False
        assert c._fine_fault == 'micro_preload_evidence_rejected:force_overload'
        assert c._fine_evidence is None and c._fine_result is None
        assert not any(f.get('stage') == 'pressure_measurement' for _,f in statuses)
        assert len(messages) == 1 and messages[0].points[0].positions[0] == .01825
        return
    _,pressure = result
    assert pressure.verified is verified
    record = next(f for _,f in statuses if f.get('stage')=='pressure_measurement')
    assert record['minimum_force'] == 6.
    assert record['retention_verified'] is False


def test_delayed_pre_epoch_samples_do_not_certify_micro_pressure(monkeypatch):
    c,node,now,_,_,statuses=pressure_fixture(monkeypatch)
    c._fine_micro_stationary_start_ns=now.nanoseconds-20_000_000
    left=tuple(ForceSample(now.nanoseconds-offset*1_000_000,2.) for offset in range(100,38,-2))
    right=tuple(ForceSample(now.nanoseconds-offset*1_000_000,2.) for offset in range(18,-1,-2))
    node._adaptive_force_histories=lambda:(c.expected_model,left,right,None)
    legacy,pressure=c._micro_pressure_snapshot(1,.01825)
    assert not legacy.verified and not pressure.verified
    record=next(f for _,f in statuses if f.get('stage')=='pressure_measurement')
    assert record['force_histories']['left']==[]
    assert len(record['raw_force_histories']['left'])==len(left)


def test_fixed_pressure_snapshot_checks_newer_endpoint_drift(monkeypatch):
    c,node,now,_,_,statuses=pressure_fixture(monkeypatch)
    node._adaptive_force_histories=lambda:(c.expected_model,(),(),None)
    set_feedback(node,now,.01825-5e-8)
    assert c._micro_pressure_snapshot(1,.01825) is None
    assert c._fine_fault=='micro_preload_pressure_endpoint_drift'
    assert not any(f.get('stage')=='pressure_measurement' for _,f in statuses)


@pytest.mark.parametrize('force', [0., .02, .049])
def test_unacquired_micro_motion_accepts_fresh_named_weak_frames_not_pressure(monkeypatch, force):
    c,node,now,_,_,_=pressure_fixture(monkeypatch)
    book=c.expected_model+'::book_base_link::collision'
    node._on_contacts(force_message(now.nanoseconds,(RIGHT,book,force)))
    assert c._observed_contact_sides()==(False,False)
    assert c._observed_contact_sides(require_minimum_force=False)==(False,True)
    assert c._feedback_error() is None
    _,pressure=c._micro_pressure_snapshot(1,.01825)
    assert not pressure.verified


@pytest.mark.parametrize('force', [0., .02])
def test_weak_named_stream_cannot_resume_the_initial_bilateral_hold(monkeypatch,force):
    c,_,now,_,messages,statuses=micro_fixture(monkeypatch)
    c._fine_side_last={'right':(now.nanoseconds,force)}
    c._wait_motion_and_stationary=lambda *args,**kwargs:True
    c._run_micro_preload()
    assert c._fine_fault=='micro_preload_contact_lost'
    assert not c._fine_micro_active
    assert not any(fields.get('stage')=='micro_preload_started' for _,fields in statuses)
    assert all(msg.points[0].positions[0]==.01825 for msg in messages)


@pytest.mark.parametrize('offset,force,reason', [
    (-150_000_001,.02,'contact_stream_lost'),
    (100_000_001,.02,'contact_stream_lost'),
    (0,float('nan'),'micro_preload_contact_lost'),
    (0,-.01,'micro_preload_contact_lost'),
])
def test_weak_micro_presence_never_accepts_stale_future_or_invalid_observations(
        monkeypatch,offset,force,reason):
    c,_,now,_,_,_=pressure_fixture(monkeypatch)
    c._fine_side_last={'right':(now.nanoseconds+offset,force)}
    assert c._feedback_error()==reason


def test_fresh_weak_pair_of_another_book_is_still_a_fault(monkeypatch):
    c,node,now,_,_,_=pressure_fixture(monkeypatch)
    node._on_contacts(force_message(now.nanoseconds,(RIGHT,BOOK,.02)))
    assert c.expected_model!='book_col_3_row_2_red'
    assert c._fine_fault=='unexpected_or_anonymous_hand_contact'


@pytest.mark.parametrize('restore_pressure', [False,True])
def test_real_micro_trajectory_pressure_dip_requires_later_six_newton_proof(
        monkeypatch,restore_pressure):
    limits=fine.FineGripLimits(maximum_force=20.,minimum_force=6.)
    c,node,now,clock,messages,statuses=fixture(monkeypatch,position=.069,limits=limits)
    def reach_first_contact(*args):
        c._fine_phase='fine'
        set_feedback(node,now,.01825)
        node._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,3.),(RIGHT,BOOK,3.)))
        return False
    c._step=reach_first_contact
    weak_frames=[]
    def sensors():
        set_feedback(node,now,c._fine_command)
        if c._fine_micro_active and c._fine_command<c._fine_micro_reference-5e-8:
            if restore_pressure and now.nanoseconds>=c._fine_motion_end_ns:
                pairs=((LEFT,BOOK,6.5),(RIGHT,BOOK,6.5))
            else:
                weak_frames.append(now.nanoseconds)
                pairs=((RIGHT,BOOK,.02),)
        else:pairs=((RIGHT,BOOK,.3),)
        node._on_contacts(force_message(now.nanoseconds,*pairs))
    clock.on_sleep=sensors
    result=c.run()
    assert result[0] is restore_pressure
    assert c._fine_fault is None and len(weak_frames)>20
    assert len(messages)==(2 if restore_pressure else 21)
    assert c._fine_stop==('micro_pressure_target_reached' if restore_pressure else 'micro_preload_limit_reached')
    records=[f for _,f in statuses if f.get('stage')=='pressure_measurement']
    assert records[-1]['pressure_evidence']['verified'] is restore_pressure
    assert all(r['retention_verified'] is False for r in records)


def test_fault_snapshot_precedes_queued_callback_and_is_emitted_once_after_hold(monkeypatch):
    c,node,now,_,messages,statuses=pressure_fixture(monkeypatch)
    c._fine_command=.0182499
    stale_stamp=now.nanoseconds-200_000_000
    c._fine_side_last={'right':(stale_stamp,.3)}
    original=node._latest_gripper_feedback
    entered,finished=threading.Event(),threading.Event()
    pending=[]
    def callback():
        entered.set()
        c.observe_contacts(force_message(now.nanoseconds,
            (RIGHT,c.expected_model+'::book_base_link::collision',.02)))
        finished.set()
    def read_feedback():
        if not pending:
            t=threading.Thread(target=callback);pending.append(t);t.start()
            assert entered.wait(1.)
            assert not finished.wait(.01)  # rejection owns the command lock
        return original()
    node._latest_gripper_feedback=read_feedback
    reason=c._feedback_error()
    assert reason=='contact_stream_lost'
    pending[0].join(timeout=1.)
    assert finished.is_set()
    assert c._fine_side_last['right']==(now.nanoseconds,.02)
    def publish(event,**fields):
        if fields.get('stage')=='fault_snapshot':
            assert messages and messages[-1].points[0].positions[0]==.01825
        statuses.append((event,fields))
    node._publish_status=publish
    c._hold(reason)
    now.nanoseconds+=2_000_000
    set_feedback(node,now,.01826)
    c._hold('later_failure')
    snapshots=[fields for _,fields in statuses if fields.get('stage')=='fault_snapshot']
    assert len(snapshots)==1
    frozen=snapshots[0]
    assert frozen['reason']=='contact_stream_lost'
    assert frozen['commanded_position']==.0182499
    assert frozen['feedback']['position']==.01825
    assert frozen['side_observations']['right']['producer_stamp_ns']==stale_stamp
    assert frozen['side_observations']['right']['force_newtons']==.3
    assert frozen['side_observations']['right']['fresh_named_pair'] is False


def test_twenty_microtargets_use_one_immutable_reference(monkeypatch):
    c,node,now,clock,messages,_=micro_fixture(monkeypatch)
    def sensors():
        set_feedback(node,now,c._fine_command)
        c._fine_side_last={'right':(now.nanoseconds,.3)}
    clock.on_sleep=sensors
    def wait(target,**kwargs):
        assert kwargs['micro']
        now.nanoseconds=max(now.nanoseconds,c._fine_motion_end_ns)
        set_feedback(node,now,target)
        c._fine_side_last={'right':(now.nanoseconds,.3)}
        return True
    c._wait_motion_and_stationary=wait
    c._micro_pressure_snapshot=lambda step,target,**kwargs:(
        SimpleNamespace(verified=True,reason='bilateral_contact_verified'),
        SimpleNamespace(verified=False,reason='left_force_below_minimum'))
    c._run_micro_preload()
    assert [msg.points[0].positions[0] for msg in messages]==[
        .01825-step*1e-7 for step in range(1,21)]
    assert c._fine_stop=='micro_preload_limit_reached'
    assert c._fine_fault is None


@pytest.mark.parametrize('maximum,minimum,measured_force', [(8.,1.,2.),(20.,6.,6.5)])
def test_real_callbacks_normal_hold_to_micro_pressure_candidate(
        monkeypatch,maximum,minimum,measured_force):
    limits = fine.FineGripLimits(maximum_force=maximum, minimum_force=minimum)
    c,node,now,clock,messages,statuses=fixture(monkeypatch,position=.069,limits=limits)
    def reach_first_contact(*args):
        c._fine_phase='fine'
        set_feedback(node,now,.01825)
        node._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,3.),(RIGHT,BOOK,3.)))
        return False
    c._step=reach_first_contact
    def sensors():
        set_feedback(node,now,c._fine_command)
        if c._fine_micro_active and c._fine_command < c._fine_micro_reference-5e-8:
            pairs=((LEFT,BOOK,measured_force),(RIGHT,BOOK,measured_force))
        else:pairs=((RIGHT,BOOK,.3),)
        node._on_contacts(force_message(now.nanoseconds,*pairs))
    clock.on_sleep=sensors
    result=c.run()
    assert result[0] is True and result[2:4]==(True,True)
    assert c._fine_stop=='micro_pressure_target_reached'
    assert c._fine_fault is None and not node._cancel.is_set()
    assert len(messages)==2  # first-bilateral measured hold, then one microstep
    assert messages[1].points[0].positions[0]==pytest.approx(.01825-1e-7)
    records=[f for _,f in statuses if f.get('stage')=='pressure_measurement']
    assert records[-1]['pressure_evidence']['verified']
    assert records[-1]['retention_verified'] is False


@pytest.mark.parametrize('missing', ['contact_publisher','gripper_subscriber'])
def test_run_requires_real_interface_readiness_before_inward_command(monkeypatch,missing):
    c,node,_,_,messages,_=fixture(monkeypatch,position=.069)
    if missing=='contact_publisher':node.count_publishers=lambda topic:0
    else:node.gripper_pub.get_subscription_count=lambda:0
    result=c.run()
    assert result[0] is False
    assert c._fine_fault
    assert all(msg.points[0].positions[0]==.069 for msg in messages)


def test_wrapper_late_fault_clears_lock_and_returns_failure(monkeypatch):
    c,node,now,_,_,_=fixture(monkeypatch)
    node.fine_gripper_close_enabled=True
    node.fine_gripper_limits=c.limits
    def run(controller):
        node._transport_lock_engaged=True
        node._adaptive_motion_started=True
        node._adaptive_overload_latched='force_overload'
        return True,.01825,True,True,True
    monkeypatch.setattr(fine.FineGripperClose,'run',run)
    result=node._adaptive_close_for_grasp()
    assert result[0] is False and node._transport_lock_engaged is False
    assert node._fine_gripper_controller is None
    assert not node._adaptive_close_active


def test_controller_exception_holds_and_disarms_before_checked_recovery(monkeypatch):
    c,node,_,_,messages,_=fixture(monkeypatch)
    node.fine_gripper_close_enabled=True
    node.fine_gripper_limits=c.limits
    def run(controller):
        raise RuntimeError('injected_controller_failure')
    monkeypatch.setattr(fine.FineGripperClose,'run',run)
    with pytest.raises(RuntimeError,match='injected_controller_failure'):
        node._adaptive_close_for_grasp()
    assert len(messages)==1 and list(messages[0].points[0].positions)==[.01825]
    assert node._fine_gripper_controller is None and not node._adaptive_close_active
    assert not node._cancel.is_set()


def test_default_disabled_profile_keeps_existing_adaptive_path(monkeypatch):
    _,node,_,_,_,_=fixture(monkeypatch)
    node.fine_gripper_close_enabled=False
    expected=(False,.01825,False,True,True)
    node._adaptive_close_attempt=lambda:expected
    assert node._adaptive_close_for_grasp()==expected
