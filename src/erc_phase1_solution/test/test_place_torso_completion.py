"""Actual torso sender and raw collectors on a bounded scalar controller model.

The model tests scheduling/admission, not a Gazebo dynamics prediction.
"""
import copy
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import numpy as np
import pytest
from ament_index_python.packages import get_package_share_directory

from erc_phase1_solution import manipulation_node as runtime
from erc_phase1_solution import place_torso_completion as torso
from erc_phase1_solution.completed_torso_hold import TorsoHoldCancellation, TorsoHoldState
from erc_phase1_solution.place_contact_guard import PlaceContactGuard
from erc_phase1_solution.scene_checked_place import SceneCheckedPlacePlan
from test_place_transition_stop import model
from test_preopen_stationary import NAMES, GOAL, MASTER, raw


def fixture(model, monkeypatch, *, start=.1, action_early=False):
    n = model.n
    n.timeout = 120.
    n._cancel = TorsoHoldCancellation()
    n._torso_hold_state = TorsoHoldState()
    n.settled_place_torso_skip_enabled = True
    n._joint_velocities = dict.fromkeys(NAMES, 0.)
    n._active_place_scene_reference = {'checked': 'scene'}
    n._selected_place_scene_reference = n._active_place_scene_reference
    n._selected_place_bin_scene = {'registered': 'bin'}
    n._selected_place_table_scene = {'registered': 'table'}
    n._place_contact_guard = PlaceContactGuard(n.now, dict(model.identity))
    n.chain = NS(lower=np.array([0., *([-3.5]*7)]), upper=np.array([.35, *([3.5]*7)]))
    first = np.asarray(GOAL); first[0] = start
    target = first.copy(); target[0] = .35
    n.joints.update(zip(torso.IK_JOINTS, first))
    plan = SceneCheckedPlacePlan([target.copy()], 0, 0., [target.copy()], [], dict(
        actual_carry_start=first.tolist(), torso_ready=target.tolist(), measured_master=MASTER,
        whole_table_pose_modeled=True, registered_bin_scene=copy.deepcopy(n._selected_place_bin_scene),
        table_scene=copy.deepcopy(n._selected_place_table_scene)))
    n._publish_status = lambda event, **data: n.events.append((event, data))
    sent, cancelled, motion = [], [], {}
    def send(goal):
        assert n._lock.locked() and n.command_lock._is_owned()
        sent.append(copy.deepcopy(goal))
        point = goal.trajectory.points[0]
        duration = point.time_from_start.sec+point.time_from_start.nanosec/1e9
        motion.update(started=n.now, initial=n.joints[torso.JOINT], target=point.positions[0],
                      duration=duration, result_after=2.2 if action_early else duration)
        result = NS(done=lambda: n.now-motion['started'] >= motion['result_after']*1e9,
                    result=lambda: NS(status=4))
        handle = NS(accepted=True, get_result_async=lambda: result)
        return NS(result=lambda: handle)
    n.torso_client = NS(wait_for_server=lambda **kw: True, send_goal_async=send)
    n.arm_client = n.head_client = object()
    n._wait_future = lambda future, timeout: future.result()
    n._cancel_goal_and_confirm = lambda *args: cancelled.append(args) or True
    n._follow = MethodType(runtime.ManipulationNode._follow, n)
    n._move_torso = MethodType(runtime.ManipulationNode._move_torso, n)
    monkeypatch.setattr(runtime, 'time', model.clock)
    monkeypatch.setattr(runtime, 'measured_scene_context', n.context)
    monkeypatch.setattr(torso, 'time', model.clock)
    monkeypatch.setattr(torso, 'measured_scene_context', n.context)
    def physical(sample):
        q = first.copy(); v = 0.
        if motion:
            elapsed = (n.now-motion['started'])/1e9
            distance = motion['target']-motion['initial']
            slope = max(-.035, min(.035, distance/motion['duration']))
            # Linear commanded ramp followed by a first-order settling tail.
            if elapsed <= motion['duration']:
                v = slope*(1-math.exp(-25*elapsed))
                q[0] = motion['initial']+slope*elapsed-slope/25*(1-math.exp(-25*elapsed))
            else:
                residual = slope/25*(1-math.exp(-25*motion['duration']))
                decay = math.exp(-25*(elapsed-motion['duration']))
                q[0] = motion['target']-residual*decay
                v = 25*residual*decay
        sample['positions'].update(zip(torso.IK_JOINTS, q))
        sample['velocities'][torso.JOINT] = v
        n._joint_velocities.update(sample['velocities'])
    n.mutate = physical
    return NS(n=n, plan=plan, identity=model.identity, sent=sent, cancelled=cancelled, motion=motion)


def execute(f):
    return torso.execute_registered_place_torso(f.n, f.plan, f.identity, get_package_share_directory)


def test_real_sender_scales_torso_duration_and_waits_for_physical_closed_endpoint(model, monkeypatch):
    f = fixture(model, monkeypatch)
    assert execute(f)
    point = f.sent[0].trajectory.points[0]
    duration_ns = point.time_from_start.sec*1_000_000_000+point.time_from_start.nanosec
    assert duration_ns == 8_928_571_429
    assert list(point.positions) == [.35] and not point.velocities and not point.accelerations
    assert f.n.joints[torso.JOINT] == pytest.approx(.35, abs=.001)
    assert abs(f.n._joint_velocities[torso.JOINT]) <= .001
    verified = [data for event, data in f.n.events if event == 'placement_torso_arrival_verified']
    assert len(verified) == 1 and 100_000_000 <= verified[0]['elapsed_ros_ns'] < 500_000_000
    assert verified[0]['accepted_samples'] >= 5
    assert not f.n._delivery_measurement_active and f.n._place_torso_arrival_owner is None
    assert not f.n._goal_handles and not f.cancelled


def test_early_controller_success_cannot_start_any_loaded_arm_or_recovery(model, monkeypatch):
    f = fixture(model, monkeypatch, action_early=True)
    arm = []
    with pytest.raises(torso.PlaceTorsoRejected, match='arrival_deadline'):
        execute(f)
        arm.append('would start arm')
    assert not arm and len(f.sent) == 1 and not f.cancelled
    rejected = [data for event, data in f.n.events if event == 'placement_torso_failed'][-1]
    assert rejected['accepted_samples'] == 0 and rejected['elapsed_ros_ns'] == 500_000_000
    assert len(rejected['recent_samples']) == 12
    assert rejected['recent_samples'][-1]['positions'][0] < .2
    assert f.n._payload_monitor_enabled and not f.n._gripper_open_confirmed
    assert not f.n._delivery_measurement_active


@pytest.mark.parametrize('fault', ['stale', 'future', 'moving', 'arm_changed', 'scene', 'attachment', 'identity', 'goal'])
def test_final_locked_torso_admission_rejects_drift_before_any_send(model, monkeypatch, fault):
    f = fixture(model, monkeypatch)
    def change(**kwargs):
        if fault == 'stale': f.n._joint_stamps_ns[torso.JOINT] = f.n.now-150_000_001
        if fault == 'future': f.n._joint_stamps_ns[torso.JOINT] = f.n.now+50_000_001
        if fault == 'moving': f.n._joint_velocities[torso.JOINT] = .00101
        if fault == 'arm_changed': f.n.joints[torso.IK_JOINTS[1]] += .00201
        if fault == 'scene': f.n._selected_place_table_scene['changed'] = True
        if fault == 'attachment': f.n._held_book_corners[0][0] += .0001
        if fault == 'identity': f.n._contact_epoch += 1
        if fault == 'goal': f.plan.setup[0][1] += .0001
        return True
    f.n.torso_client.wait_for_server = change
    with pytest.raises(torso.PlaceTorsoRejected): execute(f)
    assert not f.sent and not f.n._goal_handles
    assert not getattr(f.n, '_delivery_measurement_active', False)


def test_actual_valid_completed_hold_keeps_no_command_shortcut(model, monkeypatch):
    f = fixture(model, monkeypatch, start=.35)
    generation, _ = f.n._cancel.snapshot()
    f.n._torso_hold_state.completed = (0, generation, .35, f.n.now-1)
    assert execute(f)
    assert not f.sent
    assert any(event == 'torso_motion_skipped' for event, _ in f.n.events)
    assert not any(event == 'placement_torso_arrival_started' for event, _ in f.n.events)


@pytest.mark.parametrize('fault', ['cancel', 'contact', 'producer_reversed', 'base_moving', 'torso_velocity'])
def test_arrival_fault_holds_closed_and_cleans_its_collector(model, monkeypatch, fault):
    f = fixture(model, monkeypatch)
    physical = f.n.mutate
    def mutate(sample):
        physical(sample)
        if not getattr(f.n, '_delivery_measurement_active', False): return
        if fault == 'cancel': f.n._cancel.set()
        if fault == 'contact': f.n._payload_hazard_latched = 'contact_lost'
        if fault == 'producer_reversed' and f.n.ticks % 2 == 0:
            sample['producer_stamp_ns'] -= 50_000_000
        if fault == 'base_moving': sample['odom']['linear_speed'] = .00501
        if fault == 'torso_velocity': sample['velocities'][torso.JOINT] = .00101
    f.n.mutate = mutate
    with pytest.raises(torso.PlaceTorsoRejected): execute(f)
    assert len(f.sent) == 1 and not f.n._gripper_open_confirmed
    assert not f.n._delivery_measurement_active and f.n._place_torso_arrival_owner is None


def test_exact_archived_registered_scene_binds_without_lossy_reconstruction(model, monkeypatch):
    f = fixture(model, monkeypatch)
    p = json.loads((Path(__file__).parent/'fixtures/recorded_place_torso_scene.json').read_text())['inputs']
    f.n._active_place_scene_reference = p['selected_admission_reference']
    f.n._selected_place_scene_reference = f.n._active_place_scene_reference
    f.n._selected_place_bin_scene = p['selected_bin_scene']
    f.n._selected_place_table_scene = p['selected_table_scene']
    f.n._held_book_corners = np.asarray(p['held_book_corners'])
    f.plan.diagnostics.update(actual_carry_start=p['actual_carry_start'], torso_ready=p['torso_ready'],
        registered_bin_scene=copy.deepcopy(p['selected_bin_scene']), table_scene=copy.deepcopy(p['selected_table_scene']),
        measured_master=p['measured_master'])
    f.plan.setup = [np.asarray(p['torso_ready'])]
    f.plan.solutions = [np.asarray(p['torso_ready'])]
    owner = torso.PlaceTorsoCompletion(f.n, f.plan, f.identity, get_package_share_directory)
    np.testing.assert_array_equal(owner.start, p['actual_carry_start'])
    np.testing.assert_array_equal(owner.target, p['torso_ready'])
    assert owner.velocity_limit == .035 and owner.bin_copy == p['selected_bin_scene']


def test_actual_place_dispatches_completion_before_any_torso_arm_or_recovery_fallback(monkeypatch):
    from test_book_centered_place import place_fixture
    from test_bin_scene_admission import fixture as bin_fixture
    n, _, _ = place_fixture(True)
    n.table_scene_required = n.bin_scene_required = n.delivery_evidence_enabled = True
    n._target_book_model = 'book_col_5_row_4_blue'
    identity = dict(trial_id='trial', placement_attempt_id='attempt', target_model=n._target_book_model)
    n._selected_place_bin_scene = bin_fixture()[0]
    n._selected_place_table_scene = {}
    n._selected_place_scene_reference = object()
    n._wait_for_perception_point = lambda _: np.asarray(n._selected_place_bin_scene['floor_center'])
    n._publish_status = lambda *args, **kwargs: None
    forbidden = []
    n._move_torso = lambda *args, **kwargs: forbidden.append('unverified torso fallback')
    n._recover_closed_place = lambda *args, **kwargs: forbidden.append('recovery')
    n._execute_retained_arm_legs = lambda *args, **kwargs: forbidden.append('arm')
    q = n._carried_staging_solution.copy()
    plan = SceneCheckedPlacePlan([q.copy()], 0, 0., [q.copy()], [], {})
    monkeypatch.setattr(runtime, '_activate_place_contacts', lambda *args: None)
    monkeypatch.setattr(runtime, 'plan_scene_checked_place', lambda *args, **kwargs: plan)
    class ReachedCompletion(torso.PlaceTorsoRejected): pass
    received = []
    def completion(node, checked, correlation, resolver, **options):
        received.append((node, checked, correlation, options))
        raise ReachedCompletion('endpoint not proven')
    monkeypatch.setattr(torso, 'execute_registered_place_torso', completion)
    with pytest.raises(ReachedCompletion): n._place(identity)
    assert received == [(n, plan, identity, {'planned_start': None})]
    assert not forbidden
