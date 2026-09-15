"""Same-command pressure-epoch recovery, with fake time and real force evaluator.

No ROS node or actuator is started. The Run15 values reproduce a measured
velocity disturbance, not a claim that the synthetic continuation would retain
the physical book.
"""
from dataclasses import replace

import pytest

pytest.importorskip('rclpy')
from erc_phase1_solution import fine_gripper_close as fine
from erc_phase1_solution.adaptive_grasp import ForceSample, GripperFeedback
from test_fine_gripper_close import micro_fixture, pressure_fixture, set_feedback
from test_fine_micro_continuation import pair
from test_contact_force_timing import LEFT, RIGHT, force_message


def active_fixture(monkeypatch):
    values = pressure_fixture(monkeypatch, limits=fine.FineGripLimits(
        maximum_force=20., minimum_force=3.5, total_wall_seconds=900.))
    c,node,now,clock,_,_ = values
    c._fine_resettle_enabled = True
    c._fine_motion_end_ns = now.nanoseconds
    c._fine_step_command = c._fine_command
    c._fine_step_motion_end_ns = c._fine_motion_end_ns
    c._fine_step_last_ros_ns = now.nanoseconds
    book = c.expected_model+'::book_base_link::collision'
    def sensors():
        set_feedback(node,now,c._fine_command)
        node._on_contacts(force_message(now.nanoseconds,(LEFT,book,1.),(RIGHT,book,1.3)))
    clock.on_sleep = sensors
    sensors()
    return values


def records(statuses, stage):
    return [fields for _,fields in statuses if fields.get('stage') == stage]


def test_run15_micro4_velocity_invalidates_epoch_then_full_dwell_without_extra_close(monkeypatch):
    limits = fine.FineGripLimits(maximum_force=20.,minimum_force=3.5,total_wall_seconds=900.)
    c,node,now,clock,messages,statuses = micro_fixture(monkeypatch,limits=limits)
    reference = .018308321652810843
    c._fine_command = reference
    c._fine_first_contact = .019004878384332402
    set_feedback(node,now,reference)
    book = c.expected_model+'::book_base_link::collision'
    state = {'recheck_ns': None,'disturbed': False,'resettling': False}
    original_emit = c._emit
    def emit(stage,**fields):
        original_emit(stage,**fields)
        if stage == 'pressure_recheck_started' and fields['step'] == 4:
            state['recheck_ns'] = fields['recheck_start_ros_ns']
        if stage == 'pressure_epoch_invalidated':
            state['resettling'] = True
    c._emit = emit
    def sensors():
        q,velocity = c._fine_command,0.
        if (state['recheck_ns'] is not None and not state['disturbed']
                and now.nanoseconds-state['recheck_ns'] >= 16_000_000):
            q += 13.781069606e-9
            velocity = 6.738828029431644e-6
            state['disturbed'] = True
        set_feedback(node,now,q,velocity)
        force = (4.1,4.3) if state['resettling'] else (1.134640495,1.376372578)
        node._on_contacts(force_message(now.nanoseconds,(LEFT,book,force[0]),(RIGHT,book,force[1])))
    clock.on_sleep = sensors
    sensors()
    c._run_micro_preload(finalize=True)
    assert c._fine_fault is None and c._fine_result[0] is True
    assert state['disturbed']
    assert [m.points[0].positions[0] for m in messages] == [reference-i*1e-7 for i in range(1,5)]
    invalidated = records(statuses,'pressure_epoch_invalidated')
    assert len(invalidated) == 1 and invalidated[0]['step'] == 4
    assert invalidated[0]['reason'] == 'joint_still_moving'
    assert invalidated[0]['commanded_position'] == reference-4e-7
    endpoints = records(statuses,'stationary_endpoint')
    fresh = endpoints[-1]
    assert fresh['stationary_start_ros_ns'] > invalidated[0]['previous_stationary_start_ros_ns']
    assert fresh['stationarity_diagnostic']['longest_confirmed_stable_interval_seconds'] >= .2
    pressure = records(statuses,'pressure_measurement')[-1]
    assert pressure['stationary_start_ros_ns'] == fresh['stationary_start_ros_ns']
    assert pressure['pressure_evidence']['verified'] and pressure['retention_verified'] is False
    assert all(s['stamp_ns'] >= pressure['stationary_start_ros_ns']
               for samples in pressure['force_histories'].values() for s in samples)
    assert c._fine_micro_reference == reference


@pytest.mark.parametrize('reason', ['joint_still_moving','micro_preload_pressure_endpoint_drift'])
def test_resettle_keeps_public_target_motion_end_and_original_deadline(monkeypatch,reason):
    c,node,now,clock,messages,statuses = active_fixture(monkeypatch)
    command,end,epoch = c._fine_command,c._fine_motion_end_ns,c._fine_micro_stationary_start_ns
    set_feedback(node,now,command+30e-9,6.7e-6)
    assert c._resettle_micro_epoch(4,command,reason,wall_deadline=.3)
    assert clock.wall >= .2 and clock.wall < .3
    assert not messages and c._fine_command == command and c._fine_motion_end_ns == end
    assert c._fine_micro_stationary_start_ns > epoch and c._fine_evidence is None
    assert len(records(statuses,'pressure_epoch_invalidated')) == 1


@pytest.mark.parametrize('kind,expected', [
    ('nan','force_history_invalid'),('negative','force_history_invalid'),
    ('nonmonotonic','force_history_invalid'),('overforce','force_overload'),
    ('effort','effort_overload'),('delta','effort_delta_overload'),
])
def test_raw_hard_evidence_beats_simultaneous_motion_and_is_never_cleared(monkeypatch,kind,expected):
    c,node,now,_,messages,statuses = active_fixture(monkeypatch)
    set_feedback(node,now,c._fine_command+30e-9,6.7e-6)
    force = {'nan':float('nan'),'negative':-1.,'overforce':21.}.get(kind,1.)
    samples = (ForceSample(now.nanoseconds-10_000_000,force),ForceSample(now.nanoseconds,1.))
    if kind == 'nonmonotonic': samples = tuple(reversed(samples))
    if kind in ('effort','delta'):
        node._gripper_feedback_samples.append(GripperFeedback(now.nanoseconds,c._fine_command+30e-9,
            6.7e-6,5. if kind=='effort' else 3.5))
        node._adaptive_effort_baseline_value = 0.
    node._adaptive_force_histories = lambda:(c.expected_model,samples,samples,None)
    node._clear_target_contact_samples = lambda:pytest.fail('hard evidence was cleared')
    assert not c._resettle_micro_epoch(4,c._fine_command,'joint_still_moving',wall_deadline=5.)
    assert expected in c._fine_fault
    assert not records(statuses,'pressure_epoch_invalidated')
    assert len(messages) == 1  # permanent measured-position stop, never a close


@pytest.mark.parametrize('case,reason', [
    ('cancel','cancelled'),('identity','micro_preload_identity_lost'),
    ('stream','contact_stream_lost'),('travel','micro_preload_measured_travel_limit'),
    ('changed_target','micro_preload_command_changed'),('changed_end','micro_preload_command_changed'),
    ('before_end','joint_still_moving'),('reversed','clock_reversed'),
])
def test_hard_resume_conditions_cannot_be_softened_by_motion(monkeypatch,case,reason):
    c,node,now,_,messages,statuses = active_fixture(monkeypatch)
    set_feedback(node,now,c._fine_command,6.7e-6)
    if case == 'cancel': node._cancel.set()
    elif case == 'identity': node._target_book_model = 'book_col_1_row_1_blue'
    elif case == 'stream': c._fine_side_last = {'right':(now.nanoseconds-151_000_000,.3)}
    elif case == 'travel': set_feedback(node,now,c._fine_micro_reference-2.001e-6,6.7e-6)
    elif case == 'changed_target': c._fine_command -= 1e-7
    elif case == 'changed_end': c._fine_motion_end_ns += 1
    elif case == 'before_end':
        c._fine_motion_end_ns = c._fine_step_motion_end_ns = now.nanoseconds+1
    elif case == 'reversed': c._fine_step_last_ros_ns = now.nanoseconds+1
    error = c._micro_stationarity_error(c._fine_step_command)
    assert error == reason
    c._stop_micro(error)
    assert c._fine_fault == reason and len(messages)==1
    assert not records(statuses,'pressure_epoch_invalidated')


@pytest.mark.parametrize('bound', ['step','total'])
def test_repeated_disturbance_cannot_renew_original_step_or_total_budget(monkeypatch,bound):
    c,node,now,clock,messages,statuses = active_fixture(monkeypatch)
    original = clock.on_sleep
    def sensors():
        original()
        set_feedback(node,now,c._fine_command,6.7e-6)
    clock.on_sleep = sensors
    if bound == 'total': c._fine_wall_deadline = .03
    assert not c._resettle_micro_epoch(4,c._fine_command,'joint_still_moving',wall_deadline=.03)
    assert clock.wall <= .032 and c._fine_fault in ('step_wall_timeout','fine_grip_wall_timeout')
    assert len(messages)==1
    assert not records(statuses,'public_position_command')


@pytest.mark.parametrize('case,expected', [('stalled','clock_stalled'),
    ('expired_ros','micro_pressure_recheck_timeout'),('expired_wall','step_wall_timeout'),
    ('epoch','micro_preload_stationary_epoch_changed')])
def test_recheck_hard_chronology_wins_over_same_poll_motion(monkeypatch,case,expected):
    c,node,now,clock,messages,_ = active_fixture(monkeypatch)
    original = clock.on_sleep
    def sensors():
        original()
        if case == 'stalled': clock.wall=1.01
        elif case == 'expired_ros': now.nanoseconds+=151_000_000
        elif case == 'expired_wall': clock.wall=5.
        else: c._fine_micro_stationary_start_ns+=1
        set_feedback(node,now,c._fine_command+30e-9,6.7e-6)
        c._fine_side_last={'right':(now.nanoseconds,.3)}
    clock.on_sleep = sensors
    if case == 'stalled': clock.advancing=False
    result = c._recheck_micro_pressure(4,c._fine_command,pair(),wall_deadline=5.)
    assert result is None and c._fine_fault==expected
    assert len(messages)==1


@pytest.mark.parametrize('injection', ['initial','next_step','final'])
def test_disturbance_at_snapshot_or_admission_never_reuses_pressure_or_publishes_next_close(
        monkeypatch,injection):
    c,node,now,clock,messages,statuses = micro_fixture(monkeypatch,limits=fine.FineGripLimits(
        maximum_force=20.,minimum_force=3.5))
    book=c.expected_model+'::book_base_link::collision'
    state={'injected':False,'invalidated':False}
    def sensors():
        set_feedback(node,now,c._fine_command)
        force=4. if injection!='next_step' or state['invalidated'] else 1.
        node._on_contacts(force_message(now.nanoseconds,(LEFT,book,force),(RIGHT,book,force)))
    clock.on_sleep=sensors
    original_emit=c._emit
    def emit(stage,**fields):
        original_emit(stage,**fields)
        if stage=='pressure_epoch_invalidated':state['invalidated']=True
    c._emit=emit
    original_snapshot=c._micro_pressure_snapshot
    calls=[]
    def snapshot(*args,**kwargs):
        if kwargs.get('_resettle_validation'):return original_snapshot(*args,**kwargs)
        calls.append(args[0])
        if injection=='initial' and not state['injected']:
            state['injected']=True
            set_feedback(node,now,c._fine_command+30e-9,6.7e-6)
        result=original_snapshot(*args,**kwargs)
        # Inject between fresh evaluation and the caller's locked admission.
        wanted=2 if injection=='final' else 1
        if injection!='initial' and len(calls)==wanted and not state['injected']:
            state['injected']=True
            set_feedback(node,now,c._fine_command+30e-9,6.7e-6)
        return result
    c._micro_pressure_snapshot=snapshot
    sensors()
    c._run_micro_preload(finalize=True)
    assert state['invalidated'] and c._fine_fault is None and c._fine_result[0]
    assert not messages  # same held target, neither resettle nor admission closes
    assert len(records(statuses,'pressure_epoch_invalidated'))==1
    assert records(statuses,'pressure_measurement')[-1]['stationary_start_ros_ns'] > 10_000_000_000
