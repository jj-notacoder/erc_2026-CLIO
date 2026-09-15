"""Actual R46-derived methods: benign delivery falls back, hazards still stop.

The shared fixture uses an actual non-reentrant sensor Lock and RLock command
ownership. Its immediate action futures model the publication boundary only.
"""
from types import SimpleNamespace as NS
import math
import threading

import pytest

from test_completed_head_hold import HEAD, RIGHT, Node, normal_motion


def inject_final_delivery(node, kind='contact', extra=lambda: None):
    def hook():
        if node.hazard_calls != 2:
            return
        if kind in ('contact', 'both'):
            node._contact_generation += 1
        if kind in ('feedback', 'both'):
            node._gripper_feedback_samples.append(
                NS(position=.017, velocity=0., effort=.1, stamp_ns=node.now))
        extra()
    node.hazard_hook = hook


@pytest.mark.parametrize('kind', ['contact', 'feedback', 'both'])
def test_benign_final_delivery_runs_full_preflight_then_server_and_action(kind):
    node = Node()
    prior = node._last_completed_head_target
    order = []
    full = node._carried_head_transition_is_safe
    wait = node.head_client.wait_for_server
    send = node.head_client.send_goal_async
    node._carried_head_transition_is_safe = lambda *args: order.append('full_preflight') or full(*args)
    node.head_client.wait_for_server = lambda **kwargs: order.append('server') or wait(**kwargs)
    node.head_client.send_goal_async = lambda goal: order.append('send') or send(goal)
    inject_final_delivery(node, kind)
    normal_motion(node)
    assert order == ['full_preflight', 'server', 'send']
    assert node._last_completed_head_target != prior
    assert node._last_completed_head_target == ((0., -.6), node.now)
    assert [x[0] for x in node.geometry_calls] == ['body', 'book', 'head']


@pytest.mark.parametrize('kind', ['contact', 'feedback', 'both'])
def test_benign_delivery_never_bypasses_rejected_full_sweep(kind):
    node = Node()
    inject_final_delivery(node, kind)
    node.full_safe = False
    with pytest.raises(RuntimeError, match='blocks requested head motion'):
        node._move_head(0., -.6)
    assert node.full_preflights == [(0., -.6)]
    assert node.head_client.sends == node.head_client.waits == 0
    assert node._last_completed_head_target is None


@pytest.mark.parametrize('fault', ['cancel', 'active', 'pending', 'latched', 'sensor',
    'robot_contact', 'contact_epoch', 'attachment', 'body', 'stale', 'moving'])
@pytest.mark.parametrize('kind', ['contact', 'feedback'])
def test_mixed_delivery_and_hard_interlock_cannot_enter_fallback_or_send(fault, kind):
    node = Node()
    node.timeout = .025
    def hazard():
        if fault == 'cancel': node._cancel.set()
        elif fault == 'active': node._goal_handles.append(object())
        elif fault == 'pending': node._pending_retained_acceptances.add(object())
        elif fault == 'latched': node._payload_hazard_latched = 'contact_lost'
        elif fault == 'sensor': node._held_grip_sensor_fault = 'invalid_stock_joint_feedback'
        elif fault == 'robot_contact': node._target_robot_contact_latched = True
        elif fault == 'contact_epoch': node._contact_epoch += 1
        elif fault == 'attachment': node._held_book_corners += .001
        elif fault == 'body': node.joints[RIGHT[0]] += 2e-6
        elif fault == 'stale': node.now += 151_000_000
        elif fault == 'moving': node._joint_velocities[HEAD[0]] = .01
    inject_final_delivery(node, kind, hazard)
    if fault == 'body':
        normal_motion(node)
        return
    assert not node._move_head(0., -.6)
    assert node.full_preflights == []
    assert node.head_client.sends == node.head_client.waits == 0
    assert node._last_completed_head_target is None
    assert not any(event == 'head_motion_skipped' for event, _ in node.events)


@pytest.mark.parametrize('bad', ['missing', 'nan_position', 'nan_velocity', 'infinite_effort',
                                 'stale', 'future', 'none'])
def test_replacement_feedback_must_be_finite_and_fresh_for_fallback(bad):
    node = Node()
    node._gripper_feedback_samples.append(
        NS(position=.017, velocity=0., effort=.1, stamp_ns=node.now))
    def corrupt():
        feedback = node._gripper_feedback_samples[-1]
        if bad == 'missing': del feedback.effort
        elif bad == 'nan_position': feedback.position = math.nan
        elif bad == 'nan_velocity': feedback.velocity = math.nan
        elif bad == 'infinite_effort': feedback.effort = math.inf
        elif bad == 'stale': feedback.stamp_ns = node.now - 150_000_001
        elif bad == 'future': feedback.stamp_ns = node.now + 100_000_001
        elif bad == 'none': node._gripper_feedback_samples[-1] = None
    inject_final_delivery(node, 'feedback', corrupt)
    assert not node._move_head(0., -.6)
    assert node.full_preflights == [] and node.head_client.sends == 0


def test_target_contact_epoch_change_during_geometry_is_a_hard_stop():
    node = Node()
    node.after_geometry = lambda: setattr(node, '_contact_epoch', node._contact_epoch + 1)
    assert not node._move_head(0., -.6)
    assert node.full_preflights == [] and node.head_client.sends == 0


def test_fallback_finishes_with_real_non_reentrant_sensor_lock_and_scene_helpers():
    node = Node()
    node._active_place_scene_reference = {'fixed': True}
    node._place_contact_guard = object()
    inject_final_delivery(node, 'both')
    outcomes, failures = [], []
    def run():
        try:
            outcomes.append(node._move_head(0., -.6))
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(1.)
    assert not thread.is_alive(), 'fallback sensor/command lock deadlock'
    assert failures == [] and outcomes == [True]
    assert node.full_preflights == [(0., -.6)] and node.head_client.sends == 1
    assert node._lock.depth == node.command.depth == 0


def test_pre_send_check_is_preserved_on_fallback_with_place_command_guard():
    node = Node()
    node._place_contact_guard = object()
    inject_final_delivery(node, 'contact')
    called = []
    def stop():
        assert node.command.depth == 1 and node._lock.depth == 0
        called.append(True)
        node._cancel.set()
    assert not node._follow(node.head_client, HEAD, [0., -.6], 1.2,
                            head_preflight=True, pre_send_check=stop)
    assert called == [True]
    assert node.full_preflights == [(0., -.6)] and node.head_client.waits == 1
    assert node.head_client.sends == 0 and node._last_completed_head_target is None
