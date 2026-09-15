"""Real head/follow methods; no ROS, mesh traversal or runtime process.

Snapshot delivery is injected where real callbacks can run. The existing fixture
uses a non-reentrant sensor Lock and command RLock, and checks publication order.
"""
import ast
import hashlib
from pathlib import Path
import threading

import pytest

from test_completed_head_hold import HEAD, IK, RIGHT, Node, normal_motion


def fallback_events(node):
    return [fields for event, fields in node.events if event == 'head_hold_fallback']


@pytest.mark.parametrize('loss,stage', [(loss,stage)
    for stage in ('initial', 'geometry', 'final_helper')
    for loss in ('stale', 'moving', 'target', 'body')
    if (loss,stage) != ('body','initial')])
def test_eligibility_loss_reacquires_before_full_sweep_and_never_skips(loss, stage):
    node = Node()
    node.timeout = .2
    changed = False

    def lose():
        nonlocal changed
        changed = True
        if loss == 'stale': node._joint_stamps_ns[RIGHT[0]] = node.now - 151_000_000
        elif loss == 'moving': node._joint_velocities[IK[0]] = 1.1e-5
        elif loss == 'target': node.joints[HEAD[0]] += 2e-6
        else: node.joints[RIGHT[0]] += 2e-6

    if stage == 'initial': lose()
    elif stage == 'geometry': node.after_geometry = lose
    else:
        node.hazard_hook = lambda: lose() if node.hazard_calls == 2 else None

    publish = node._publish_status
    def observe(event, **fields):
        assert node._lock.depth == 0
        publish(event, **fields)
        if event == 'head_hold_fallback' and fields.get('phase') == 'waiting_for_fresh_stationary_start':
            assert node.command.depth == 0, 'wait must not hold command lock and block contact callbacks'
            assert changed
            # A new ordinary sensor update restores eligibility; measured
            # positions keep their small change so this must be a full sweep.
            with node._lock:
                node._joint_stamps_ns = {n:node.now for n in node.joints}
                node._joint_velocities = {n:0. for n in node.joints}
    node._publish_status = observe
    normal_motion(node)
    assert [row['phase'] for row in fallback_events(node)] == [
        'waiting_for_fresh_stationary_start', 'fresh_start_ready']
    assert fallback_events(node)[-1]['producer_stamps_ns'][RIGHT[0]] == node.now


@pytest.mark.parametrize('loss', ['stale', 'moving', 'missing', 'nonfinite'])
def test_no_fresh_stationary_delivery_expires_with_reason_without_sweep(loss):
    node = Node()
    node.timeout = .025
    if loss == 'stale': node._joint_stamps_ns[RIGHT[0]] = node.now - 151_000_000
    elif loss == 'moving': node._joint_velocities[RIGHT[0]] = .01
    elif loss == 'missing': del node.joints[RIGHT[0]]
    else: node.joints[RIGHT[0]] = float('nan')
    assert not node._move_head(0., -.6)
    assert node.full_preflights == [] and node.head_client.sends == 0
    assert node.events[-1][1]['reason'] == 'fresh_stationary_start_timeout'
    assert node.events[-1][1]['attempts'] >= 2
    assert node._last_completed_head_target is None


@pytest.mark.parametrize('fault', ['cancel', 'active', 'pending', 'epoch', 'attachment',
                                  'latched', 'sensor', 'robot', 'contact', 'scene', 'retention'])
def test_independent_hard_stop_wins_even_when_fresh_snapshot_is_reacquired(fault):
    node = Node()
    node._joint_stamps_ns[RIGHT[0]] = node.now - 151_000_000
    node._active_place_scene_reference = {'fixed': True}
    publish = node._publish_status
    def deliver(event, **fields):
        publish(event, **fields)
        if event != 'head_hold_fallback' or fields.get('phase') != 'waiting_for_fresh_stationary_start':
            return
        with node._lock:
            node._joint_stamps_ns = {n:node.now for n in node.joints}
            if fault == 'cancel': node._cancel.set()
            elif fault == 'active': node._goal_handles.append(object())
            elif fault == 'pending': node._pending_retained_acceptances.add(object())
            elif fault == 'epoch': node._contact_epoch += 1
            elif fault == 'attachment': node._held_book_corners += .001
            elif fault == 'latched': node._payload_hazard_latched = 'contact_lost'
            elif fault == 'sensor': node._held_grip_sensor_fault = 'invalid_stock_joint_feedback'
            elif fault == 'robot': node._target_robot_contact_latched = True
            elif fault == 'contact': node.place_fault = 'arm_bin'
            elif fault == 'scene': node.scene_fault = 'base_moved'
            else: node.retention_fault = 'contact_lost'
    node._publish_status = deliver
    if fault in ('contact', 'scene'):
        with pytest.raises(RuntimeError, match='placement_'): node._move_head(0., -.6)
    else:
        assert not node._move_head(0., -.6)
    assert node.full_preflights == [] and node.head_client.sends == 0
    assert not any(e == 'head_motion_skipped' for e, _ in node.events)


def test_original_sweep_rejection_after_reacquisition_still_stops_before_server():
    node = Node()
    node.after_geometry = lambda: node.joints.__setitem__(RIGHT[0], 2e-6)
    node.full_safe = False
    with pytest.raises(RuntimeError, match='blocks requested head motion'):
        node._move_head(0., -.6)
    assert node.full_preflights == [(0., -.6)]
    assert node.head_client.waits == node.head_client.sends == 0


def test_new_diagnostic_publisher_failure_does_not_change_safe_fallback_result():
    node = Node()
    node.after_geometry = lambda: node.joints.__setitem__(RIGHT[0], 2e-6)
    node._publish_status = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('diagnostic unavailable'))
    assert node._move_head(0., -.6)
    assert node.full_preflights == [(0., -.6)] and node.head_client.sends == 1


def test_reacquisition_releases_both_locks_for_callback_thread_and_joins():
    node = Node()
    node.timeout = .3
    node._joint_velocities[IK[0]] = 1.1e-5
    waiting = threading.Event()
    publish = node._publish_status
    def observe(event, **fields):
        publish(event, **fields)
        if event == 'head_hold_fallback': waiting.set()
    node._publish_status = observe
    def callback():
        assert waiting.wait(.5)
        # Use the actual locks, not the fixture's single-thread depth counters.
        with node.command.raw:
            with node._lock.raw:
                node._joint_velocities[IK[0]] = 0.
                node._joint_stamps_ns = {n:node.now for n in node.joints}
    worker = threading.Thread(target=callback, daemon=True)
    worker.start()
    normal_motion(node)
    worker.join(.5)
    assert not worker.is_alive()
    assert node._lock.depth == node.command.depth == 0


def test_stationary_body_pose_present_before_geometry_uses_new_current_geometry():
    node = Node()
    node.joints[RIGHT[0]] = .02
    assert node._move_head(0., -.6)
    assert node.geometry_calls[0][2]['right_positions'][0] == .02
    assert node.head_client.sends == 0 and node.events[-1][0] == 'head_motion_skipped'


def test_torso_pre_send_check_and_cancellation_hook_remain_active():
    node = Node()
    hook = []
    def check():
        hook.append('ran')
        node._cancel.set()
    assert not node._follow(node.head_client, [IK[0]], [.35], 2.2, pre_send_check=check)
    assert hook == ['ran'] and node.head_client.sends == 0
