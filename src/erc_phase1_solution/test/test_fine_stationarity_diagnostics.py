"""Telemetry must explain waits without changing the frozen Run10 decisions."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip('rclpy')
from sensor_msgs.msg import JointState
from erc_phase1_solution import fine_gripper_close as fine
from erc_phase1_solution.adaptive_grasp import ForceSample, GripperFeedback
from test_fine_gripper_close import pressure_fixture, set_feedback


FIXTURES=Path(__file__).parent/'fixtures'


def run_wait(monkeypatch, schedule, *, baseline=False, seconds=.6):
    c,node,now,clock,messages,statuses=pressure_fixture(monkeypatch)
    c._fine_motion_end_ns=now.nanoseconds
    c._fine_phase='micro_preload'
    node._adaptive_motion_started=True
    initial=now.nanoseconds
    target=c._fine_command
    def sensors():
        ms=round((now.nanoseconds-initial)/1_000_000)
        stamp=now.nanoseconds
        q,v=target,0.
        if schedule=='velocity_spikes' and ms in (50,100): v=2.1e-6
        elif schedule=='position_spikes' and ms in (50,100): q+=26e-9
        elif schedule=='future_stored': stamp+=2_000_000
        elif schedule=='repeated_resets' and ms%40<2: v=2.1e-6
        elif schedule=='duplicates':
            if ms%10: return
        elif schedule=='callback_future':
            message=JointState()
            message.header.stamp.sec,message.header.stamp.nanosec=divmod(stamp+2_000_000,10**9)
            message.name=['gripper_left_finger_joint']
            message.position,message.velocity,message.effort=[q],[v],[.1]
            node._on_joint_state(message)
            c._fine_side_last={'right':(now.nanoseconds,.3)}
            return
        set_feedback(node,now,q,velocity=v,stamp=stamp)
        c._fine_side_last={'right':(now.nanoseconds,.3)}
    sensors()
    clock.on_sleep=sensors
    if baseline:
        spec=importlib.util.spec_from_file_location('run10_frozen_wait',FIXTURES/'run10_wait_method.py')
        old=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        old.time=clock
        outcome=old._wait_motion_and_stationary(c,target,micro=True,wall_deadline=seconds)
    else:
        outcome=c._wait_motion_and_stationary(target,micro=True,wall_deadline=seconds)
    return c,node,now,clock,messages,statuses,outcome


@pytest.mark.parametrize('schedule', ['steady','velocity_spikes','position_spikes',
                                    'future_stored','duplicates','callback_future','repeated_resets'])
def test_diagnostics_preserve_run10_wait_decisions_and_command_timing(monkeypatch,schedule):
    old=run_wait(monkeypatch,schedule,baseline=True)
    new=run_wait(monkeypatch,schedule)
    assert new[-1] is old[-1]
    assert new[0]._fine_fault==old[0]._fine_fault
    assert new[3].wall==old[3].wall
    assert new[0]._fine_micro_stationary_start_ns==old[0]._fine_micro_stationary_start_ns
    assert [list(m.points[0].positions) for m in new[4]]==[list(m.points[0].positions) for m in old[4]]


def terminal_diagnostic(statuses,success):
    if success:
        rows=[f['stationarity_diagnostic'] for _,f in statuses if f.get('stage')=='stationary_endpoint']
    else:
        rows=[f['diagnostic'] for _,f in statuses if f.get('stage')=='stationarity_diagnostic']
    assert len(rows)==1
    return rows[0]


@pytest.mark.parametrize('schedule,reason', [('velocity_spikes','velocity'),('position_spikes','position_error')])
def test_real_dwell_resets_are_counted_once_per_transition(monkeypatch,schedule,reason):
    *_,statuses,success=run_wait(monkeypatch,schedule)
    assert success
    diag=terminal_diagnostic(statuses,True)
    assert diag['stationary_resets']==2
    assert diag['resets_by_reason'][reason]==2
    assert diag['nonstationary_new_feedback_by_reason'][reason]==2
    assert diag['longest_confirmed_stable_interval_seconds']>=.2
    assert 'observed_feedback_samples' not in diag and 'observed_reset_samples' not in diag


def test_production_joint_callback_clamps_small_future_stamps_before_wait(monkeypatch):
    c,node,now,_,_,statuses,success=run_wait(monkeypatch,'callback_future')
    assert success and node._latest_gripper_feedback().stamp_ns<=now.nanoseconds
    diag=terminal_diagnostic(statuses,True)
    assert diag['max_future_lag_ns']==0
    assert diag['nonstationary_new_feedback_by_reason'].get('future_feedback',0)==0


def test_duplicate_polls_do_not_inflate_distinct_stored_feedback_counts(monkeypatch):
    *_,statuses,success=run_wait(monkeypatch,'duplicates')
    assert success
    diag=terminal_diagnostic(statuses,True)
    assert diag['distinct_feedback_stamps']==21
    assert diag['duplicate_feedback_polls']==80


def test_failure_rings_are_bounded_and_emitted_after_the_existing_hold(monkeypatch):
    c,node,now,clock,messages,statuses,success=run_wait(monkeypatch,'repeated_resets',seconds=3.)
    assert not success and c._fine_fault=='step_wall_timeout'
    diag=terminal_diagnostic(statuses,False)
    assert len(diag['observed_feedback_samples'])==256
    assert len(diag['observed_reset_samples'])==32
    assert diag['stationary_resets']>32 and diag['distinct_feedback_stamps']>256
    assert diag['resets_by_reason']['velocity']==diag['stationary_resets']
    stages=[f.get('stage') for _,f in statuses]
    assert stages.index('fault_snapshot')<stages.index('stationarity_diagnostic')
    assert len(messages)==1 and messages[0].points[0].positions[0]==c._fine_command
    assert clock.wall<=3.002


def test_future_stored_feedback_is_reported_as_observed_not_claimed_raw_producer(monkeypatch):
    *_,statuses,success=run_wait(monkeypatch,'future_stored')
    assert not success
    diag=terminal_diagnostic(statuses,False)
    assert diag['max_future_lag_ns']==2_000_000
    assert diag['nonstationary_new_feedback_by_reason']['future_feedback']>0
    assert diag['stationary_resets']==0  # never stationary; not a stable-to-unstable reset


@pytest.mark.parametrize('minimum,expected', [(3.,True),(3.5,True),(6.,False)])
def test_recorded_run10_pressure_needs_the_recheck_span_even_at_lower_target(monkeypatch,minimum,expected):
    limits=fine.FineGripLimits(maximum_force=20.,minimum_force=minimum)
    c,node,now,_,messages,_=pressure_fixture(monkeypatch,limits=limits)
    payloads=json.loads((FIXTURES/'run10_pressure_step2.json').read_text(encoding='utf-8-sig'))['payloads']
    outcomes=[]
    for p in payloads:
        now.nanoseconds=p['evaluation_ros_ns']
        c.expected_model=node._target_book_model=p['expected_model']
        c._fine_command=p['commanded_position']
        c._fine_micro_reference=p['reference_position']
        c._fine_micro_stationary_start_ns=p['stationary_start_ros_ns']
        c._fine_motion_end_ns=p['stationary_start_ros_ns']
        node._gripper_feedback_samples.append(GripperFeedback(**p['feedback']))
        histories={side:tuple(ForceSample(**s) for s in rows) for side,rows in p['force_histories'].items()}
        node._adaptive_force_histories=lambda:(c.expected_model,histories['left'],histories['right'],None)
        c._fine_side_last={side:(rows[-1].stamp_ns,rows[-1].force_newtons) for side,rows in histories.items()}
        outcomes.append(c._micro_pressure_snapshot(2,c._fine_command)[1])
    assert not outcomes[0].verified
    assert outcomes[1].verified is expected
    if expected:
        assert outcomes[0].reason=='force_sample_span_too_short'
        assert outcomes[1].left_samples==outcomes[1].right_samples==37
    assert not messages and c._fine_fault is None
