"""Run28/29 endpoint readings are measured data, never an out-of-range public command."""
from dataclasses import replace
from types import SimpleNamespace
import math
import pytest
pytest.importorskip('rclpy')
from erc_phase1_solution import stock_gripper_close as stock
from erc_phase1_solution import lift_pressure_gate as gates
from erc_phase1_solution.adaptive_grasp import ForceSample
from test_stock_gripper_close import selected, close_fixture

MASTER = 'gripper_left_finger_joint'
RECORDED_RUN28_Q = -3.7475186295953393e-13
RECORDED_Q = -2.735730263894206e-9  # Run29 before_initial_shelf_lift


@pytest.mark.parametrize('q,valid', [
    (RECORDED_RUN28_Q, True), (RECORDED_Q, True),
    (-1e-6, True), (math.nextafter(-1e-6, -math.inf), False),
    (.069 + 1e-6, True), (math.nextafter(.069 + 1e-6, math.inf), False),
    (-2e-6, False), (.070, False), (math.nan, False), (math.inf, False)])
def test_stock_measured_range_is_finite_and_bounded(q, valid):
    assert stock.measured_position_in_range(q) is valid


@pytest.mark.parametrize('position', [RECORDED_RUN28_Q, RECORDED_Q])
@pytest.mark.parametrize('supported,left,right,accepted', [
    (True, True, False, True), (True, False, True, False),
    (False, True, False, False), (False, True, True, True)])
def test_recorded_fresh_probe_preserves_contact_mode(monkeypatch, position, supported, left, right, accepted):
    _, n, now, _, _, events, joint, _ = selected(monkeypatch)
    n._gravity_supported_payload = supported
    n.joints[MASTER] = position
    n._on_joint_state(joint())
    n.grasp_contact_max_age = .4
    n._left_target_contact_ns = now.nanoseconds if left else 0
    n._right_target_contact_ns = now.nanoseconds if right else 0
    def fresh_contacts(seconds):
        assert seconds == .20 and n._retention_probe_active
        assert n._left_target_contact_ns == n._right_target_contact_ns == 0
        n._left_target_contact_ns = now.nanoseconds if left else 0
        n._right_target_contact_ns = now.nanoseconds if right else 0
        return True
    n._wait_sim_duration = fresh_contacts
    command, phase = (('compact_transport', 'post_retreat_cradle_roll') if supported
                      else ('pick', 'before_initial_shelf_lift'))
    assert n._fresh_retention_probe(command, phase) is accepted
    assert not n._retention_probe_active
    fields = next(fields for event, fields in events if event == 'retention_verified')
    assert fields['measured_position'] == position
    assert fields['plausible_width'] is True
    assert fields['left_contact'] is left and fields['right_contact'] is right
    if accepted:
        assert not any(event == 'grasp_lost' for event, _ in events)


@pytest.mark.parametrize('stock_enabled', [False, True])
def test_actual_feedback_preserves_raw_q_and_default_strict_boundary(monkeypatch, stock_enabled):
    _, n, _, _, _, _, joint, _ = selected(monkeypatch)
    n.stock_gripper_close_diagnostic_enabled = stock_enabled
    n.joints[MASTER] = RECORDED_Q
    n._on_joint_state(joint())
    feedback, error = n._adaptive_motion_feedback()
    if stock_enabled:
        assert error is None and feedback.position == RECORDED_Q
        assert n._pinch_sample()[0]
    else:
        assert feedback is None and error == 'invalid_gripper_motion_feedback'
        assert not n._pinch_sample()[0]
    assert n.joints[MASTER] == RECORDED_Q


@pytest.mark.parametrize('q,command', [(RECORDED_Q, 0.), (.069+5e-7, .069), (.018, .018), (-2e-6, None)])
def test_cancel_hold_uses_only_valid_public_endpoint_and_logs_raw(monkeypatch, q, command):
    _, n, _, _, sent, events, joint, _ = selected(monkeypatch)
    n.gripper_pub = SimpleNamespace(publish=sent.append)
    n.joints[MASTER] = q
    n._on_joint_state(joint())
    n._hold_adaptive_gripper('cancelled')
    n._hold_adaptive_gripper('cancelled')
    if command is None:
        assert sent == []
        assert events[-1][1]['feedback_reason'] == 'invalid_gripper_motion_feedback'
    else:
        assert len(sent) == 1
        assert list(sent[0].points[0].positions) == [command]
        assert 0 <= command <= .069
        assert events[-1][1]['measured_position'] == q
        assert events[-1][1]['commanded_hold_position'] == command


@pytest.mark.parametrize('q', [RECORDED_Q, .069+5e-7])
def test_roundoff_is_not_allowed_for_requested_commands(monkeypatch, q):
    _, n, _, _, sent, _, _, _ = selected(monkeypatch)
    n.gripper_pub = SimpleNamespace(publish=sent.append)
    assert not n._publish_adaptive_gripper_position(q, motion_seconds=1., wait_seconds=1.2)
    assert n._adaptive_motion_halt_reason == 'invalid_adaptive_position_command'
    assert len(sent) == 1  # Valid measured safety hold, never the requested target.
    assert list(sent[0].points[0].positions) == [.0183]


def test_stock_evidence_acquisition_metadata_and_live_gate_use_same_raw_bound(monkeypatch):
    gate, n, now, _, sent, events, joint, send = selected(monkeypatch)
    n.joints[MASTER] = RECORDED_Q
    n._on_joint_state(joint())
    feedback = n._latest_gripper_feedback()
    samples = [ForceSample(now.nanoseconds-d, 0.) for d in (60_000_000, 30_000_000, 0)]
    proof = stock.contact_evidence(samples, samples, feedback, now.nanoseconds)
    assert proof.verified and proof.width == RECORDED_Q
    result = n._publish_adaptive_close_result(proof, verified=True, reason='recorded_roundoff',
        close_stage='stock_public_position', commanded_position=0., acquisition_position=RECORDED_Q)
    assert result[0]
    assert events[-1][1]['acquisition_plausible_width'] is True
    gate.checked['fingers'][MASTER] = 0.
    assert gate.send(send) == 'accepted-future'
    assert len(sent) == 1
    assert n.joints[MASTER] == RECORDED_Q


@pytest.mark.parametrize('bad', ['range', 'velocity', 'stale', 'contact'])
def test_stock_contact_evidence_roundoff_does_not_skip_other_guards(monkeypatch, bad):
    _, n, now, _, _, _, _, _ = selected(monkeypatch)
    feedback = replace(n._latest_gripper_feedback(), position=RECORDED_Q)
    samples = [ForceSample(now.nanoseconds-d, 1.) for d in (60_000_000, 30_000_000, 0)]
    if bad == 'range': feedback = replace(feedback, position=-2e-6)
    elif bad == 'velocity': feedback = replace(feedback, velocity=.0031)
    elif bad == 'stale': feedback = replace(feedback, stamp_ns=now.nanoseconds-151_000_000)
    assert not stock.contact_evidence(samples, [] if bad == 'contact' else samples,
                                      feedback, now.nanoseconds).verified


def test_final_gate_rejects_measured_value_outside_roundoff_allowance(monkeypatch):
    gate, n, _, _, sent, _, joint, send = selected(monkeypatch)
    gate.checked['fingers'][MASTER] = 0.
    n.joints[MASTER] = -2e-6
    n._on_joint_state(joint())
    with pytest.raises(gates.LiftPressureRejected, match='width_invalid'):
        gate.send(send)
    assert not sent


def test_stock_close_success_keeps_q_zero_command_when_measured_q_rounds_negative(monkeypatch):
    controller, n, _, _, sent, _, sensors = close_fixture(monkeypatch)
    n.joints[MASTER] = RECORDED_Q
    sensors()
    result = controller.run()
    assert result[0] and result[1] == RECORDED_Q
    assert len(sent) == 1 and list(sent[0].points[0].positions) == [0.]
    assert not n._adaptive_hold_sent
