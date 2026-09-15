"""Real contact callbacks and public messages, with deterministic fake clocks.

These tests exercise the stopped coarse-to-fine transition, not physical grip
retention. No ROS node, simulator or controller is started.
"""
from dataclasses import asdict, replace
import threading

import pytest

pytest.importorskip('rclpy')
from erc_phase1_solution import fine_gripper_close as fine
from test_fine_gripper_close import fixture, set_feedback
from test_contact_force_timing import BOOK, LEFT, RIGHT, force_message


def coarse_fixture(monkeypatch, *, position=.01906115835798718, limits=None):
    c,node,now,clock,messages,statuses = fixture(monkeypatch, position=position, limits=limits)
    c._fine_phase = 'coarse'
    c._fine_command = .019
    node._on_contacts(force_message(now.nanoseconds, (RIGHT, BOOK, .527)))
    assert c._fine_stop == 'coarse_contact_observed'
    return c,node,now,clock,messages,statuses


def fresh_held_sensors(c,node,now):
    set_feedback(node,now,c._fine_command)
    node._on_contacts(force_message(now.nanoseconds,(RIGHT,BOOK,.3)))


def test_first_coarse_contact_immediately_holds_measured_position_once(monkeypatch):
    c,node,now,_,messages,statuses = coarse_fixture(monkeypatch)
    assert c._fine_fault is None and node._adaptive_motion_halt_reason is None
    assert node._target_book_model is None
    assert c.expected_model == 'book_col_3_row_2_red'
    assert c._fine_coarse_contact_model == c.expected_model
    assert [m.points[0].positions[0] for m in messages] == [.01906115835798718]
    assert messages[0].points[0].time_from_start.nanosec == 20_000_000
    node._on_contacts(force_message(now.nanoseconds+2_000_000,(RIGHT,BOOK,.6)))
    with c._fine_lock:
        assert not c._publish_once_locked(.019,.32,'close')
    assert len(messages) == 1
    assert not any(f.get('acquired') for _,f in statuses)


@pytest.mark.parametrize('other', ['anonymous_book::collision',
                                 'book_col_3_row_2_blue::book_base_link::collision'])
def test_untrusted_coarse_contact_cannot_create_a_resumable_stop(monkeypatch,other):
    c,node,now,_,_,_=fixture(monkeypatch,position=.021)
    c._fine_phase='coarse'
    node._on_contacts(force_message(now.nanoseconds,(RIGHT,other,.3)))
    assert c._fine_fault=='unexpected_or_anonymous_hand_contact'
    assert c._fine_stop==c._fine_fault and c._fine_coarse_contact_model is None
    assert not c._resume_coarse_contact()


def test_simultaneous_bilateral_coarse_contact_uses_existing_bilateral_path(monkeypatch):
    c,node,now,_,messages,_=fixture(monkeypatch,position=.021)
    c._fine_phase='coarse'
    node._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,.3),(RIGHT,BOOK,.4)))
    assert c._fine_stop=='bilateral_contact_observed' and c._fine_fault is None
    assert c._fine_coarse_contact_model is None
    assert node._target_book_model==c.expected_model
    assert len(messages)==1 and messages[0].points[0].positions[0]==.021
    assert not c._resume_coarse_contact()


def test_handoff_requires_full_post_hold_dwell_then_bounded_fine_command(monkeypatch):
    c,node,now,clock,messages,statuses = coarse_fixture(monkeypatch)
    held = c._fine_command
    motion_end = c._fine_motion_end_ns
    clock.on_sleep = lambda:fresh_held_sensors(c,node,now)
    assert c._resume_coarse_contact()
    endpoint = next(f for _,f in statuses if f.get('stage')=='stationary_endpoint')
    assert endpoint['stationary_start_ros_ns'] >= motion_end
    assert now.nanoseconds-endpoint['stationary_start_ros_ns'] >= 200_000_000
    assert clock.wall >= .22 and len(messages)==1
    assert c._fine_stop is None and c._fine_fault is None
    assert c._fine_phase == 'fine' and c._fine_first_contact == held
    assert c._fine_evidence is None and node._target_book_model is None
    assert c._step(held-5e-6,'fine')
    assert len(messages)==2 and messages[-1].points[0].positions[0]==held-5e-6
    handoff = next(f for _,f in statuses if f.get('stage')=='coarse_contact_handoff')
    assert handoff['acquired'] is False


@pytest.mark.parametrize('bad_target', [.01906115835798718-5.001e-6,
                                      .01906115835798718+1e-9])
def test_handoff_rejects_larger_closure_or_reopening_before_publication(monkeypatch,bad_target):
    c,node,now,clock,messages,_ = coarse_fixture(monkeypatch)
    clock.on_sleep=lambda:fresh_held_sensors(c,node,now)
    assert c._resume_coarse_contact()
    assert not c._step(bad_target,'fine')
    assert c._fine_fault=='coarse_contact_handoff_step_limit'
    assert all(m.points[0].positions[0]==.01906115835798718 for m in messages)


def test_no_coarse_command_can_be_sent_after_successful_handoff(monkeypatch):
    c,node,now,clock,messages,_ = coarse_fixture(monkeypatch)
    clock.on_sleep=lambda:fresh_held_sensors(c,node,now)
    assert c._resume_coarse_contact()
    assert not c._step(.019,'coarse')
    assert c._fine_fault=='coarse_command_after_contact'
    assert all(m.points[0].positions[0]==.01906115835798718 for m in messages)


@pytest.mark.parametrize('case', [
    'stale_contact','missing_contact','nonfinite_contact','negative_contact',
    'future_contact','changed_identity','changed_latch','cancelled',
    'overload','effort_overload','effort_delta_overload','stale_feedback',
    'missing_feedback','travel','wall_timeout',
])
def test_handoff_cannot_clear_any_live_guard_failure(monkeypatch,case):
    c,node,now,_,messages,statuses = coarse_fixture(monkeypatch)
    if case=='stale_contact': c._fine_side_last={'right':(now.nanoseconds-150_000_001,.3)}
    elif case=='missing_contact': c._fine_side_last={}
    elif case=='nonfinite_contact': c._fine_side_last={'right':(now.nanoseconds,float('nan'))}
    elif case=='negative_contact': c._fine_side_last={'right':(now.nanoseconds,-.1)}
    elif case=='future_contact': c._fine_side_last={'right':(now.nanoseconds+100_000_001,.3)}
    elif case=='changed_identity': c.expected_model='book_col_2_row_4_red'
    elif case=='changed_latch': node._target_book_model='book_col_2_row_4_red'
    elif case=='cancelled': node._cancel.set()
    elif case=='overload': node._adaptive_overload_latched='force_overload'
    elif case in ('effort_overload','effort_delta_overload'):
        node._adaptive_overload_latched=case
    elif case=='stale_feedback': set_feedback(node,now,c._fine_command,stamp=now.nanoseconds-150_000_001)
    elif case=='missing_feedback': node._gripper_feedback_samples.clear()
    elif case=='travel': c._fine_first_contact=c._fine_command+.003500001
    elif case=='wall_timeout': c._fine_wall_deadline=0.
    assert not c._resume_coarse_contact()
    assert c._fine_fault and c._fine_stop==c._fine_fault
    assert not any(f.get('stage')=='coarse_contact_handoff' for _,f in statuses)
    assert all(m.points[0].positions[0]>=.01906115835798718-1e-12 for m in messages)


@pytest.mark.parametrize('case', ['clock_stalled','clock_reversed','position_error','velocity'])
def test_hold_dwell_remains_bounded_and_stationary(monkeypatch,case):
    c,node,now,clock,messages,_ = coarse_fixture(monkeypatch)
    def sensors():
        if case=='clock_reversed': now.nanoseconds-=4_000_000
        set_feedback(node,now,c._fine_command+(1e-6 if case=='position_error' else 0.),
                     velocity=3e-6 if case=='velocity' else 0.)
        node._on_contacts(force_message(now.nanoseconds,(RIGHT,BOOK,.3)))
    clock.on_sleep=sensors
    clock.advancing=case!='clock_stalled'
    assert not c._resume_coarse_contact()
    assert c._fine_fault==({'clock_stalled':'clock_stalled','clock_reversed':'clock_reversed'}
                          .get(case,'step_wall_timeout'))
    assert clock.wall<=5.01
    assert not any(m.points[0].positions[0]<.01906115835798718 for m in messages)


@pytest.mark.parametrize('when', ['during_dwell','after_dwell'])
def test_bilateral_arrival_upgrades_normal_hold_without_resuming_fine(monkeypatch,when):
    c,node,now,clock,messages,statuses = coarse_fixture(monkeypatch)
    def bilateral():
        set_feedback(node,now,c._fine_command)
        node._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,.4),(RIGHT,BOOK,.5)))
    if when=='during_dwell': clock.on_sleep=bilateral
    else:
        def waited(*a,**kw): bilateral(); return True
        c._wait_motion_and_stationary=waited
    assert not c._resume_coarse_contact()
    assert c._fine_stop=='bilateral_contact_observed' and c._fine_fault is None
    assert node._target_book_model==c.expected_model
    assert len(messages)==2
    assert not any(f.get('stage')=='coarse_contact_handoff' for _,f in statuses)
    assert c._fine_evidence is None


@pytest.mark.parametrize('pair,force', [
    (RIGHT,21.), ('robot::gripper_left_palm_link::collision',.3),
])
def test_fault_during_dwell_permanently_overrides_coarse_hold(monkeypatch,pair,force):
    c,node,now,clock,messages,_=coarse_fixture(monkeypatch)
    def sensors():
        set_feedback(node,now,c._fine_command)
        node._on_contacts(force_message(now.nanoseconds,(pair,BOOK,force)))
    clock.on_sleep=sensors
    assert not c._resume_coarse_contact()
    assert c._fine_fault in ('force_overload','unexpected_target_palm_contact')
    node._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,1.),(RIGHT,BOOK,1.)))
    assert c._fine_stop==c._fine_fault
    assert all(m.points[0].positions[0]==.01906115835798718 for m in messages)


def test_new_feedback_after_final_dwell_sample_cannot_authorize_handoff(monkeypatch):
    c,node,now,_,messages,_=coarse_fixture(monkeypatch)
    def waited(*a,**kw):
        now.nanoseconds=c._fine_motion_end_ns+200_000_000
        fresh_held_sensors(c,node,now)
        set_feedback(node,now,c._fine_command-1e-6)
        return True
    c._wait_motion_and_stationary=waited
    assert not c._resume_coarse_contact()
    assert c._fine_fault=='coarse_contact_endpoint_changed'
    assert not any(m.points[0].positions[0]<.01906115835798718-1.001e-6 for m in messages)


def test_queued_overload_wins_between_handoff_and_next_publication(monkeypatch):
    c,node,now,clock,messages,_=coarse_fixture(monkeypatch)
    clock.on_sleep=lambda:fresh_held_sensors(c,node,now)
    assert c._resume_coarse_contact()
    entered=threading.Event()
    original=node._interrupt_adaptive_gripper_if_fault
    def interrupt(): entered.set(); return original()
    node._interrupt_adaptive_gripper_if_fault=interrupt
    with c._fine_lock:
        thread=threading.Thread(target=lambda:node._on_contacts(
            force_message(now.nanoseconds,(RIGHT,BOOK,21.))))
        thread.start()
        assert entered.wait(1.)
        assert node._adaptive_overload_reason()=='force_overload'
        assert not c._publish_once_locked(c._fine_command-5e-6,.2,'close')
    thread.join(1.)
    assert not thread.is_alive() and c._fine_fault=='force_overload'
    assert all(m.points[0].positions[0]==.01906115835798718 for m in messages)


def test_run_switches_to_five_micrometre_steps_above_nominal_fine_start(monkeypatch):
    c,node,now,clock,messages,statuses=fixture(monkeypatch,position=.069)
    touched=False
    def sensors():
        nonlocal touched
        if c._fine_phase=='coarse' and not touched:
            touched=True
            set_feedback(node,now,.024061)
        else:set_feedback(node,now,c._fine_command)
        if touched:
            node._on_contacts(force_message(now.nanoseconds,(RIGHT,BOOK,.3)))
        if c._fine_phase=='fine' and c._fine_command<.024061-1e-9:
            node._cancel.set()
    clock.on_sleep=sensors
    result=c.run()
    assert result[0] is False and c._fine_fault=='cancelled'
    commands=[f for _,f in statuses if f.get('stage')=='public_position_command']
    coarse=[f for f in commands if f['purpose']=='close' and f['phase']=='coarse']
    refined=[f for f in commands if f['purpose']=='close' and f['phase']=='fine']
    assert len(coarse)==len(refined)==1
    assert coarse[0]['position']==.024
    assert refined[0]['position']==pytest.approx(.024061-5e-6)
    assert any(f.get('stage')=='coarse_contact_handoff' for _,f in statuses)
    assert not any(f.get('stage')=='micro_preload_started' for _,f in statuses)
    assert node._transport_lock_engaged is False


def test_long_diagnostic_budget_is_opt_in_and_does_not_change_mechanical_bounds():
    defaults=fine.FineGripLimits()
    extended=replace(defaults,total_wall_seconds=900.)
    assert defaults.total_wall_seconds==300.
    old,new=asdict(defaults),asdict(extended)
    assert new.pop('total_wall_seconds')==900.
    old.pop('total_wall_seconds')
    assert new==old
    assert extended.step_wall_seconds==5. and extended.preclose_wall_seconds==10.
    with pytest.raises(ValueError): replace(defaults,total_wall_seconds=1500.00001)


def test_extended_diagnostic_budget_still_stops_at_its_total_deadline(monkeypatch):
    c,node,now,clock,_,_=coarse_fixture(monkeypatch,limits=fine.FineGripLimits(total_wall_seconds=900.))
    clock.wall=899.9
    clock.on_sleep=lambda:fresh_held_sensors(c,node,now)
    assert not c._resume_coarse_contact()
    # The existing wait clips its step deadline to the overall deadline, then
    # reports its ordinary step timeout. It must not obtain another 5 seconds.
    assert c._fine_fault=='step_wall_timeout'
    assert clock.wall<=900.003
