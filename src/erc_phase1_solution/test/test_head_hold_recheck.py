"""Actual head/follow methods; collision spies delimit software-only coverage."""
from types import SimpleNamespace as NS

import pytest

from test_completed_head_hold import HEAD, IK, RIGHT, Node, normal_motion


def recover(node, initial='stale_body', on_ready=lambda: None):
    if initial == 'stale_head': node._joint_stamps_ns[HEAD[0]] = node.now-151_000_000
    elif initial == 'stale_body': node._joint_stamps_ns[RIGHT[0]] = node.now-151_000_000
    elif initial == 'moving_head': node._joint_velocities[HEAD[0]] = 1.1e-4
    else: node._joint_velocities[IK[0]] = 1.1e-5
    publish = node._publish_status
    restored = False
    def receive(event, **fields):
        nonlocal restored
        publish(event, **fields)
        if (not restored and event == 'head_hold_fallback'
                and fields.get('phase') == 'waiting_for_fresh_stationary_start'):
            assert node._lock.depth == node.command.depth == 0
            restored = True
            with node._lock:
                node._joint_stamps_ns = {n:node.now for n in node.joints}
                node._joint_velocities = {n:0. for n in node.joints}
            on_ready()
    node._publish_status = receive


def enabled():
    node = Node()
    node.completed_head_hold_recheck_enabled = True
    return node


def phases(node):
    return [f.get('phase') for e,f in node.events if e == 'head_hold_fallback']


def fault(node, name):
    if name == 'cancel': node._cancel.set()
    elif name == 'active': node._goal_handles.append(object())
    elif name == 'pending': node._pending_retained_acceptances.add(object())
    elif name == 'epoch': node._contact_epoch += 1
    elif name == 'attachment': node._held_book_corners += .001
    elif name == 'latched': node._payload_hazard_latched = 'contact_lost'
    elif name == 'sensor': node._held_grip_sensor_fault = 'invalid_stock_joint_feedback'
    elif name == 'robot': node._target_robot_contact_latched = True
    elif name == 'contact': node.place_fault = 'arm_bin'
    elif name == 'scene': node.scene_fault = 'base_moved'
    else: node.retention_fault = 'contact_lost'


@pytest.mark.parametrize('initial', ['stale_head','stale_body','moving_head','moving_torso'])
def test_one_initial_recovery_requires_original_geometry_and_final_locked_proof(initial):
    node = enabled();completed = node._last_completed_head_target
    recover(node, initial)
    assert node._move_head(0., -.6)
    assert node.head_client.sends == node.head_client.waits == 0 and node.full_preflights == []
    assert [row[0] for row in node.geometry_calls] == ['body','book','head']
    assert phases(node) == ['waiting_for_fresh_stationary_start','fresh_start_ready','strict_target_recheck_ready']
    assert node._last_completed_head_target == completed
    assert node.events[-1][0] == 'head_motion_skipped'
    assert node.events[-1][1]['completed_action_ros_ns'] == completed[1]
    assert node.events[-1][1]['maximum_joint_change'] <= 1e-6
    assert node._lock.depth == node.command.depth == 0


@pytest.mark.parametrize('flag', [False, None, 1, 'true'])
def test_default_or_non_boolean_optin_preserves_original_full_action(flag):
    node = Node();node.completed_head_hold_recheck_enabled = flag
    recover(node)
    normal_motion(node)
    assert 'strict_target_recheck_ready' not in phases(node) and node.geometry_calls == []


def test_absent_flag_preserves_original_full_action():
    node = Node();recover(node)
    normal_motion(node)
    assert node.geometry_calls == []


@pytest.mark.parametrize('offset', [1.01e-6,-1.01e-6,.01])
def test_fresh_start_without_exact_target_cannot_admit_a_hold(offset):
    node = enabled();node.joints[HEAD[0]] += offset;recover(node)
    normal_motion(node)
    assert 'fresh_start_ready' in phases(node)
    assert 'strict_target_recheck_ready' not in phases(node) and node.geometry_calls == []


def test_changed_requested_target_still_requires_original_action():
    node = enabled()
    normal_motion(node, pan=.5e-6)
    assert node.geometry_calls == [] and phases(node) == []


@pytest.mark.parametrize('stage', ['waiting','geometry'])
@pytest.mark.parametrize('name', ['cancel','active','pending','epoch','attachment','latched','sensor','robot','contact','scene','retention'])
def test_hard_faults_win_before_recheck_admission_and_cannot_send(stage, name):
    node = enabled();node._active_place_scene_reference = {'fixed':True}
    recover(node, on_ready=(lambda: fault(node,name)) if stage=='waiting' else lambda: None)
    if stage == 'geometry': node.after_geometry = lambda: fault(node,name)
    if name in ('contact','scene'):
        with pytest.raises(RuntimeError, match='placement_'): node._move_head(0., -.6)
    else:
        assert not node._move_head(0., -.6)
    assert node.head_client.sends == node.head_client.waits == 0 and node.full_preflights == []
    assert not any(e=='head_motion_skipped' for e,_ in node.events)
    assert node._lock.depth == node.command.depth == 0


@pytest.mark.parametrize('which', ['body','book','head'])
def test_each_original_current_geometry_predicate_can_reject_retry(which):
    node = enabled();recover(node);setattr(node,which+'_collision','collision')
    with pytest.raises(RuntimeError, match='Current head hold geometry is unsafe'):
        node._move_head(0., -.6)
    assert node.head_client.sends == node.head_client.waits == 0 and node.full_preflights == []
    assert not any(e=='head_motion_skipped' for e,_ in node.events)


@pytest.mark.parametrize('joint', [RIGHT[0], HEAD[0]])
@pytest.mark.parametrize('safe', [True,False])
def test_final_geometry_drift_uses_full_original_sweep_without_second_retry(joint, safe):
    node=enabled();recover(node);node.full_safe=safe
    node.after_geometry=lambda: node.joints.__setitem__(joint,node.joints[joint]+2e-6)
    if safe: normal_motion(node)
    else:
        with pytest.raises(RuntimeError, match='blocks requested head motion'): node._move_head(0., -.6)
        assert node.head_client.sends == node.head_client.waits == 0
    assert node.full_preflights == [(0.,-.6)]
    assert phases(node).count('strict_target_recheck_ready') == 1
    assert [r[0] for r in node.geometry_calls] == ['body','book','head']


@pytest.mark.parametrize('kind', ['contact','feedback','both'])
def test_final_delivery_change_retains_original_fallback_with_no_recursive_retry(kind):
    node=enabled();recover(node)
    def deliver():
        if node.hazard_calls != 3:return
        if kind in ('contact','both'):node._contact_generation += 1
        if kind in ('feedback','both'):
            node._gripper_feedback_samples.append(NS(position=.017,velocity=0.,effort=.1,stamp_ns=node.now))
    node.hazard_hook=deliver
    normal_motion(node)
    assert phases(node).count('strict_target_recheck_ready') == 1
    assert [r[0] for r in node.geometry_calls] == ['body','book','head']


class Clock:
    def __init__(self):self.now=0.;self.sleeps=[]
    def monotonic(self):return self.now
    def sleep(self,seconds):self.sleeps.append(seconds);self.now += seconds


@pytest.mark.parametrize('stage', ['before_geometry','after_geometry'])
def test_retry_cannot_admit_after_original_waiting_window(monkeypatch, stage):
    node=enabled();node.timeout=.25;clock=Clock()
    monkeypatch.setitem(Node._try_completed_head_hold.__globals__,'time',clock)
    recover(node)
    if stage=='before_geometry':
        publish=node._publish_status
        def expire(event, **fields):
            publish(event,**fields)
            if fields.get('phase')=='strict_target_recheck_ready':clock.now=.26
        node._publish_status=expire
    else:node.after_geometry=lambda: setattr(clock,'now',.26)
    normal_motion(node)
    assert any(f.get('reason')=='recheck_window_expired' for _,f in node.events)
    assert phases(node).count('strict_target_recheck_ready') == 1
    assert len(node.geometry_calls)==(0 if stage=='before_geometry' else 3)


def test_retry_failure_does_not_restart_original_deadline(monkeypatch):
    node=enabled();node.timeout=.25;clock=Clock()
    monkeypatch.setitem(Node._try_completed_head_hold.__globals__,'time',clock)
    recover(node)
    def expire_feedback():
        clock.now=.20;node._joint_stamps_ns[RIGHT[0]]=node.now-151_000_000
    node.after_geometry=expire_feedback
    assert not node._move_head(0.,-.6)
    assert .25 <= clock.now < .28 and len(clock.sleeps)==3
    assert node.events[-1][1]['reason']=='fresh_stationary_start_timeout'
    assert node.head_client.sends==node.head_client.waits==0 and node.full_preflights==[]
    assert phases(node).count('strict_target_recheck_ready')==1


@pytest.mark.parametrize('kind', ['nonfinite','stale','none'])
def test_expired_retry_window_never_masks_original_invalid_feedback_fault(monkeypatch, kind):
    node=enabled();node.timeout=.25;clock=Clock()
    node._gripper_feedback_samples.append(NS(position=.017,velocity=0.,effort=.1,stamp_ns=node.now))
    monkeypatch.setitem(Node._try_completed_head_hold.__globals__,'time',clock)
    recover(node);node.after_geometry=lambda: setattr(clock,'now',.26)
    def bad_delivery():
        if node.hazard_calls != 3:return
        value=NS(position=.017,velocity=0.,effort=.1,stamp_ns=node.now)
        if kind=='nonfinite':value.effort=float('nan')
        elif kind=='stale':value.stamp_ns=node.now-150_000_001
        else:value=None
        node._gripper_feedback_samples.append(value)
    node.hazard_hook=bad_delivery
    assert not node._move_head(0.,-.6)
    assert node.events[-1][1]['reason']=='invalid_retained_feedback'
    assert node.head_client.sends==node.head_client.waits==0 and node.full_preflights==[]
    assert not any(e=='head_motion_skipped' for e,_ in node.events)
