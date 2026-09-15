"""Pure extraction controller tests: no ROS, publishers, simulation or poses."""
from dataclasses import replace
import math

import pytest

from erc_phase1_solution.adaptive_grasp import GripperFeedback
from erc_phase1_solution.extraction_pressure import (
    ArmFeedback, ExtractionPressureLimits, ExtractionPressureMaintenance,
    ExtractionSnapshot, NamedForceSample,
)

MODEL = 'book_col_3_row_2_red'
Q = .01830
ARM = (0.,)*8
ORIGIN = 10_000_000_000


def controller(**kwargs):
    return ExtractionPressureMaintenance(
        expected_model=MODEL, acquired_position=Q,
        acquisition_reference_position=kwargs.pop('reference', Q+.0000005),
        acquisition_microsteps_used=kwargs.pop('used', 5),
        arm_endpoint=ARM, started_ros_ns=ORIGIN, started_wall_seconds=50.,
        **kwargs)


def snapshot(t, *, q=Q, force=4., right=None, arm=ARM, gripper_velocity=0.,
             arm_velocity=0., lead_ns=-2_000_000, wall_factor=1., **kwargs):
    now = ORIGIN+round(t*1e9)
    latest = now+lead_ns
    def samples(value):
        return tuple(NamedForceSample(MODEL, latest-(63-i)*2_000_000, value) for i in range(64))
    return ExtractionSnapshot(
        now, 50.+t*wall_factor, MODEL,
        GripperFeedback(latest, q, gripper_velocity, 0.),
        ArmFeedback(latest, tuple(arm), (arm_velocity,)*8),
        samples(force), samples(force if right is None else right), True,
        **kwargs)


def settle(c, *, start=0., force=4., arm=ARM, lead_ns=-2_000_000):
    for i in range(31):
        t = round(start+i*.02, 9)
        decision = c.observe(snapshot(t, q=c.target, force=force, arm=arm, lead_ns=lead_ns))
        if decision.action in ('READY_FOR_ARM', 'CLOSE_STEP', 'STOP', 'PREFIX_VERIFIED'):
            return t, decision
    pytest.fail('bounded pure fixture did not settle')


def correction(c, decision, *, force=4., arm=ARM):
    t0 = (decision.snapshot.now_ns-ORIGIN)/1e9
    previous = c.target
    issued = c.authorize_correction(decision.token, decision.snapshot, live=decision.snapshot)
    assert issued.action == 'CLOSE_STEP'
    for i in range(1, 36):
        elapsed = i*.02
        fraction = min(1., elapsed/.20)
        q = previous+fraction*(issued.position-previous)
        decision = c.observe(snapshot(t0+elapsed, q=q, force=force, arm=arm,
                                       gripper_velocity=(issued.position-previous)/.20 if fraction < 1 else 0.))
        if decision.action in ('READY_FOR_ARM', 'CLOSE_STEP', 'STOP'):
            return round(t0+elapsed, 9), decision
    pytest.fail('correction never reached bounded endpoint decision')


def test_fresh_stationary_pressure_is_needed_before_first_arm_permit():
    c = controller()
    t, decision = settle(c)
    assert t >= .22
    assert decision.action == 'READY_FOR_ARM'
    assert decision.evidence.verified
    assert c.corrections == 0
    assert c.arm_started(decision.token, decision.snapshot, live=decision.snapshot, distance=.001).action == 'ARM_PERMIT'
    assert c.phase == 'arm_running'


def test_weak_pressure_rechecks_same_q_then_one_bounded_correction():
    c = controller()
    t, decision = settle(c, force=.2)
    assert t >= .28
    assert decision.action == 'CLOSE_STEP'
    assert decision.position == pytest.approx(Q-.0000001, abs=1e-15)
    assert c.corrections == 0  # A request does not publish or consume travel.
    end, ready = correction(c, decision)
    assert end >= t+.4
    assert ready.action == 'READY_FOR_ARM'
    assert c.corrections == 1


def test_no_early_dwell_while_tiny_gripper_trajectory_still_interpolates():
    c = controller()
    t, step = settle(c, force=.2)
    c.authorize_correction(step.token, step.snapshot, live=step.snapshot)
    assert c.observe(snapshot(t+.18, q=c.target, force=4.)).action == 'WAIT'
    assert c._stationary_epoch is None


def test_no_correction_or_new_arm_permission_while_arm_is_running():
    c = controller()
    t, decision = settle(c)
    c.arm_started(decision.token, decision.snapshot, live=decision.snapshot, distance=.001)
    weak = c.observe(snapshot(t+.02, force=.1, arm_velocity=.02))
    assert weak.action == 'MONITOR'
    assert c.corrections == 0


def test_pressure_does_not_authorize_an_arm_endpoint_before_fresh_post_arm_dwell():
    c = controller()
    t, ready = settle(c)
    c.arm_started(ready.token, ready.snapshot, live=ready.snapshot, distance=.001)
    endpoint = (0., .001, 0., 0., 0., 0., 0., 0.)
    finished = snapshot(t+.10, arm=endpoint)
    assert c.arm_finished(ready.token, finished, endpoint=endpoint,
                          terminal_confirmed=True, succeeded=True).action == 'WAIT'
    t2, next_ready = settle(c, start=t+.12, arm=endpoint)
    assert next_ready.action == 'READY_FOR_ARM'
    assert t2 >= t+.32
    assert c.extraction_distance == pytest.approx(.001)


@pytest.mark.parametrize('failure', ['contact_lost', 'payload_robot_contact', 'cancelled', 'payload_transform_ambiguous'])
def test_hazard_is_permanent_even_if_contacts_later_look_good(failure):
    c = controller()
    s = snapshot(.02, hazard=failure)
    assert c.observe(s).action == 'STOP'
    assert c.observe(snapshot(.04)).reason == failure
    assert c.corrections == 0


def test_even_an_empty_external_failure_message_is_a_permanent_stop():
    c = controller()
    assert c.stop('').reason == 'unspecified_stop'
    assert c.observe(snapshot(.02)).action == 'STOP'


@pytest.mark.parametrize('side', ['left', 'right'])
def test_missing_named_side_never_enters_reacquisition(side):
    c = controller()
    assert c.observe(replace(snapshot(.02), **{side: ()})).reason == 'contact_lost:'+side
    assert c.observe(snapshot(.04)).action == 'STOP'


@pytest.mark.parametrize('change,reason', [
    ({'model': 'book_col_3_row_3_red'}, 'identity_changed'),
    ({'transform_valid': False}, 'payload_transform_ambiguous'),
    ({'transform_valid': None}, 'payload_transform_ambiguous'),
])
def test_identity_and_transform_invalidation(change, reason):
    assert controller().observe(replace(snapshot(.02), **change)).reason == reason


@pytest.mark.parametrize('mutation,reason', [
    (lambda s: replace(s, gripper=replace(s.gripper, effort=4.01)), 'effort_overload'),
    (lambda s: replace(s, gripper=replace(s.gripper, position=Q-.000003)), 'measured_travel_or_width_limit'),
    (lambda s: replace(s, gripper=replace(s.gripper, velocity=math.nan)), 'invalid_gripper_feedback'),
    (lambda s: replace(s, arm=replace(s.arm, velocities=(math.nan,)*8)), 'invalid_arm_feedback'),
    (lambda s: replace(s, left=s.left[:-1]+(replace(s.left[-1], force_newtons=20.01),)), 'force_overload'),
    (lambda s: replace(s, left=s.left[:-1]+(replace(s.left[-1], model='book_col_9_row_2_red'),)), 'unexpected_contact_identity'),
    (lambda s: replace(s, left=s.left+(s.left[-1],)), 'invalid_force_chronology'),
])
def test_loaded_interlocks_reject_before_any_command(mutation, reason):
    c = controller()
    assert c.observe(mutation(snapshot(.02))).reason == reason
    assert c.corrections == 0


def test_effort_delta_guard_remains_active():
    c = controller(baseline_effort=-1.)
    s = snapshot(.02)
    assert c.observe(replace(s, gripper=replace(s.gripper, effort=2.01))).reason == 'effort_delta_overload'


def test_unavailable_effort_is_not_fabricated_as_positive_pressure():
    c = controller()
    s = snapshot(.02)
    assert c.observe(replace(s, gripper=replace(s.gripper, effort=math.nan))).action == 'WAIT'


def test_clock_reverse_stall_and_wall_budget_are_terminal():
    assert controller().observe(snapshot(-.01)).reason == 'clock_reversed'
    c = controller()
    c.observe(snapshot(.02))
    assert c.observe(replace(snapshot(.02), wall_seconds=51.03)).reason == 'clock_stalled'
    assert controller().observe(replace(snapshot(.02), wall_seconds=171.)).reason == 'total_wall_timeout'


def test_stale_side_is_hard_stop_despite_other_side_and_joint_being_fresh():
    c = controller()
    s = snapshot(.02)
    assert c.observe(replace(s, left=snapshot(-.2).left)).reason == 'contact_or_feedback_stale'


def test_fixed_future_snapshot_can_catch_up_without_certifying_future_data():
    c = controller()
    t, decision = settle(c, lead_ns=2_000_000)
    assert decision.action == 'READY_FOR_ARM'
    fixed = decision.snapshot
    assert max(fixed.arm.stamp_ns, fixed.gripper.stamp_ns,
               fixed.left[-1].stamp_ns, fixed.right[-1].stamp_ns) <= fixed.now_ns
    assert c.arm_started(decision.token, fixed, live=fixed, distance=.001).action == 'ARM_PERMIT'


def test_physical_outlier_during_future_catchup_resets_stationary_proof():
    c = controller()
    for i in range(1, 9):
        c.observe(snapshot(i*.02, lead_ns=2_000_000))
    outlier = c.observe(snapshot(.18, q=Q+.000001, lead_ns=2_000_000))
    assert outlier.action == 'WAIT'
    for i in range(10, 19):
        assert c.observe(snapshot(i*.02, lead_ns=2_000_000)).action == 'WAIT'


def test_observation_gap_cannot_count_as_continuous_stationarity():
    c = controller()
    for i in range(5):
        c.observe(snapshot(i*.02))
    assert c.observe(snapshot(.20)).action == 'WAIT'
    for i in range(11, 20):
        assert c.observe(snapshot(i*.02)).action == 'WAIT'


def test_single_use_permission_rejects_reuse_and_new_hazards():
    c = controller()
    _, ready = settle(c)
    assert c.arm_started(ready.token, ready.snapshot, live=ready.snapshot, distance=.001).action == 'ARM_PERMIT'
    assert c.arm_started(ready.token, ready.snapshot, live=ready.snapshot, distance=.001).reason == 'stale_or_reused_permission'
    c = controller()
    t, ready = settle(c)
    c.observe(snapshot(t+.02, hazard='contact_lost'))
    assert c.arm_started(ready.token, ready.snapshot, live=ready.snapshot, distance=.001).reason == 'contact_lost'


def test_new_feedback_epoch_invalidates_previous_ready_permission():
    c = controller()
    t, ready = settle(c)
    newer = c.observe(snapshot(t+.02))
    assert newer.action == 'READY_FOR_ARM' and newer.token != ready.token
    assert c.arm_started(ready.token, newer.snapshot, live=newer.snapshot, distance=.001).reason == 'stale_or_reused_permission'


def test_pressure_recovery_before_command_cancels_unneeded_close_request():
    c = controller()
    t, step = settle(c, force=.2)
    # A complete fresh strong window supersedes the unconsumed weak request.
    for i in range(1, 5):
        result = c.observe(snapshot(t+i*.02))
    assert result.action == 'READY_FOR_ARM'
    assert c.corrections == 0
    assert c.authorize_correction(step.token, result.snapshot, live=result.snapshot).action == 'STOP'


def test_remaining_original_acquisition_budget_cannot_be_reset():
    c = controller(reference=Q+.00000195)
    _, result = settle(c, force=.2)
    assert result.reason == 'cumulative_correction_budget_exhausted'
    assert c.corrections == 0


def test_correction_budget_is_shared_across_arm_segments():
    c = controller()
    _, step = settle(c, force=.2)
    t, ready = correction(c, step)
    first_target = c.target
    c.arm_started(ready.token, ready.snapshot, live=ready.snapshot, distance=.001)
    endpoint = (0., .001, 0., 0., 0., 0., 0., 0.)
    s = snapshot(t+.1, q=c.target, arm=endpoint)
    c.arm_finished(ready.token, s, endpoint=endpoint, terminal_confirmed=True, succeeded=True)
    _, second = settle(c, start=t+.12, force=.2, arm=endpoint)
    assert second.action == 'CLOSE_STEP'
    assert second.position == pytest.approx(Q-.0000002, abs=1e-15)
    assert second.position == pytest.approx(first_target-.0000001, abs=1e-15)
    assert c.corrections == 1


def test_unconfirmed_arm_stop_never_requests_pressure_correction():
    c = controller()
    t, ready = settle(c)
    c.arm_started(ready.token, ready.snapshot, live=ready.snapshot, distance=.001)
    stop = c.arm_finished(ready.token, snapshot(t+.1, force=.1), endpoint=ARM,
                         terminal_confirmed=False, succeeded=False)
    assert stop.reason == 'unconfirmed_or_failed_arm_segment'
    assert c.observe(snapshot(t+.12)).action == 'STOP'


def test_arm_distance_and_arm_wall_timeout_are_bounded():
    c = controller()
    _, ready = settle(c)
    assert c.arm_started(ready.token, ready.snapshot, live=ready.snapshot, distance=.002).reason == 'extraction_distance_limit'
    c = controller()
    t, ready = settle(c)
    c.arm_started(ready.token, ready.snapshot, live=ready.snapshot, distance=.001)
    assert c.observe(snapshot(t+5.01)).reason == 'arm_segment_wall_timeout'


def test_acquisition_commands_consume_combined_command_budget_even_with_travel_left():
    c = controller(used=20)
    _, result = settle(c, force=.2)
    assert result.reason == 'cumulative_correction_budget_exhausted'
    assert c.corrections == 0


def test_only_one_remaining_acquisition_command_can_be_used():
    c = controller(used=19, reference=Q+.0000019)
    _, step = settle(c, force=.2)
    assert step.action == 'CLOSE_STEP'
    _, terminal = correction(c, step, force=.2)
    assert terminal.reason == 'cumulative_correction_budget_exhausted'
    assert c.corrections == 1


def test_consuming_old_permit_ages_its_evidence_against_fresh_clocks():
    c = controller()
    t, ready = settle(c)
    result = c.arm_started(ready.token, ready.snapshot, distance=.001,
                           live=snapshot(t+.2))
    assert result.reason == 'contact_or_feedback_stale'


def test_new_live_pressure_deficiency_revokes_previously_valid_arm_permit():
    c = controller()
    t, ready = settle(c)
    assert c.arm_started(ready.token, ready.snapshot, distance=.001,
                          live=snapshot(t+.02, force=.1)).reason == 'pressure_permission_invalidated'


def test_new_live_fault_is_checked_before_close_publication_permission():
    c = controller()
    t, step = settle(c, force=.2)
    assert c.authorize_correction(step.token, step.snapshot,
        live=snapshot(t+.02, force=20.01)).reason == 'force_overload'
    assert c.corrections == 0


def test_superseded_correction_cannot_close_against_recovered_pressure():
    c = controller()
    t, step = settle(c, force=.2)
    assert c.authorize_correction(step.token, step.snapshot,
        live=snapshot(t+.02)).reason == 'pressure_correction_superseded'
    assert c.corrections == 0


@pytest.mark.parametrize('field', ['positions', 'velocities'])
def test_malformed_arm_arrays_stop_instead_of_escaping_as_exceptions(field):
    s = snapshot(.02)
    s = replace(s, arm=replace(s.arm, **{field: None}))
    assert controller().observe(s).reason == 'invalid_arm_feedback'


def test_verified_prefix_has_no_additional_arm_permission_or_full_delivery_claim():
    c = controller()
    t, ready = settle(c)
    endpoint = ARM
    for segment in range(5):
        assert ready.action == 'READY_FOR_ARM'
        assert c.arm_started(ready.token, ready.snapshot, live=ready.snapshot,
                              distance=.001).action == 'ARM_PERMIT'
        endpoint = (0., (segment+1)*.001, 0., 0., 0., 0., 0., 0.)
        t = round(t+.10, 9)
        assert c.arm_finished(ready.token, snapshot(t, arm=endpoint), endpoint=endpoint,
            terminal_confirmed=True, succeeded=True).action == 'WAIT'
        t, ready = settle(c, start=t+.02, arm=endpoint)
    assert ready.action == 'PREFIX_VERIFIED'
    assert ready.reason == 'bounded_prefix_only'
    assert ready.token is None
    assert c.extraction_distance == pytest.approx(.005)


@pytest.mark.parametrize('kwargs', [
    {'maximum_force': 21.}, {'correction_step': .0000002}, {'maximum_corrections': 21},
    {'global_preload_travel': .000003}, {'dwell_seconds': .1}, {'maximum_age_seconds': .2},
    {'maximum_effort': 4.1}, {'maximum_segment_distance': .002}, {'maximum_corrections': True},
])
def test_configuration_cannot_exceed_bounded_policy(kwargs):
    with pytest.raises(ValueError):
        ExtractionPressureLimits(**kwargs)
