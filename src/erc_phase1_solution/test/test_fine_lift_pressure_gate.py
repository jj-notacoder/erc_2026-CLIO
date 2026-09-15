"""Fresh pressure -> single arm admission, with actual ROS sensor callbacks."""
from collections import deque
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
from sensor_msgs.msg import JointState
from erc_phase1_solution import lift_pressure_gate as module
from erc_phase1_solution.adaptive_grasp import ForceSample, GripperFeedback
from erc_phase1_solution.manipulation_node import ManipulationNode, RetainedMotionNotStopped
from erc_phase1_solution.motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS
from test_contact_force_timing import contact_node, force_message, LEFT, RIGHT, BOOK, MODEL


def fixture(monkeypatch):
    node, now = contact_node()
    node._adaptive_close_active = False
    node._adaptive_gripper_command_lock = threading.RLock()
    node._lock = threading.Lock()
    node._cancel = threading.Event()
    node._goal_handles = []
    node._pending_retained_acceptances = set()
    node._held_book_corners = None
    node._transport_lock_engaged = True
    node._acquired_effort_baseline = -.1
    node.adaptive_contact_force_maximum = 20.
    node.adaptive_contact_samples = 3
    node.adaptive_contact_max_age = .15
    node.adaptive_contact_max_gap = .075
    node.adaptive_contact_min_span = .05
    node.adaptive_contact_max_skew = .05
    node.adaptive_effort_maximum = 4.
    node.adaptive_effort_delta_maximum = 3.
    node.grasp_min_position = .0155
    node.grasp_min_margin = .0005
    node.grasp_max_position = .05
    node.target_colour = 'red'
    node.lift_first_minimum_contact_force = 3.5
    node.joints.update({name: .0 for name in (*IK_JOINTS, *RIGHT_ARM_JOINTS, 'head_1_joint', 'head_2_joint')})
    node.joints['torso_lift_joint'] = .35
    node.joints['gripper_left_finger_joint'] = .0183
    node._staging_odom = dict(stamp_ns=now.nanoseconds, pose=(2., -.1, .02), linear_speed=0., angular_speed=0.)
    events, sent = [], []
    node._publish_status = lambda event, **fields: events.append((event, fields))
    wall = SimpleNamespace(seconds=100., hook=None)

    def joint_message(stamp=None):
        message = JointState()
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp or now.nanoseconds, 10**9)
        message.name = list(node.joints)
        message.position = list(node.joints.values())
        message.velocity = [0.] * len(message.name)
        message.effort = [-.1] * len(message.name)
        return message

    node._on_joint_state(joint_message())
    for stamp in range(now.nanoseconds-60_000_000, now.nanoseconds+1, 2_000_000):
        # Independent callbacks are the official topology; never fabricate a
        # simultaneous (left,right) tuple by treating one missing side as zero.
        node._on_contacts(force_message(stamp, (LEFT, BOOK, 4.)))
        node._on_contacts(force_message(stamp, (RIGHT, BOOK, 4.2)))
    checked = dict(left=[node.joints[name] for name in IK_JOINTS],
        fingers={'gripper_left_finger_joint': .0183},
        reported_passive_joints=[], modeled_finger_joints=['gripper_left_outer_knuckle_left_joint'],
        geometry_context=dict(right_positions=(0.,)*7, head_positions=(0.,)*2))
    reference = dict(joints=dict(node.joints), base_pose=(2., -.1, .02))
    def sleep(seconds):
        wall.seconds += seconds
        now.nanoseconds += round(seconds * 1e9)
        if wall.hook:
            wall.hook()
    monkeypatch.setattr(module.time, 'monotonic', lambda: wall.seconds)
    monkeypatch.setattr(module.time, 'sleep', sleep)
    gate = module.LiftPressureGate(node, checked, reference)
    def send():
        assert node._lock.locked()
        sent.append(now.nanoseconds)
        return 'accepted-future'
    return gate, node, now, wall, sent, events, joint_message, send


def test_actual_callbacks_and_velocities_admit_exactly_once(monkeypatch):
    gate, node, now, wall, sent, events, _, send = fixture(monkeypatch)
    assert gate.send(send) == 'accepted-future'
    assert len(sent) == 1
    assert node._joint_velocities['arm_left_1_joint'] == 0.
    assert events[-1][0] == 'lift_first_pressure_admitted'
    with pytest.raises(module.LiftPressureRejected, match='already_consumed'):
        gate.send(send)
    assert len(sent) == 1


@pytest.mark.parametrize('change,reason', [
    ('weak', 'weak'), ('force', 'force_overload'), ('effort', 'effort_overload'),
    ('delta', 'effort_delta_overload'), ('cancel', 'cancelled'),
    ('identity', 'target_identity_invalid'), ('anonymous', 'unexpected_contact_identity'),
    ('stale', 'stale'), ('arm_velocity', 'arm_not_stationary'),
    ('missing_arm_velocity', 'arm_feedback_invalid'), ('gripper_velocity', 'gripper_not_stationary'),
    ('right_context', 'collision_context_changed'), ('head_reference', 'collision_context_changed'),
    ('base', 'registered_base_moved'), ('finger', 'finger_geometry_changed'),
    ('hazard', 'contact_lost'), ('width', 'width_invalid'),
    ('baseline', 'acquisition_effort_baseline_missing'), ('pending', 'arm_motion_pending'),
])
def test_no_motion_for_invalid_fresh_admission(monkeypatch, change, reason):
    gate, n, now, _, sent, _, _, send = fixture(monkeypatch)
    if change in ('weak', 'force'):
        n._book_contact_force_samples[MODEL][0].append(ForceSample(now.nanoseconds+1, .24 if change=='weak' else 21.))
    elif change in ('effort', 'delta', 'gripper_velocity', 'width'):
        values = {'effort': dict(effort=5.), 'delta': dict(effort=3.2),
                  'gripper_velocity': dict(velocity=2.1e-6), 'width': dict(position=.014)}
        n._gripper_feedback_samples.append(replace(n._gripper_feedback_samples[-1], **values[change]))
    elif change == 'cancel': n._cancel.set()
    elif change == 'identity': n._target_book_model = 'book_col_4_row_1_red'
    elif change == 'anonymous': n._book_contact_force_samples[''] = ([ForceSample(now.nanoseconds,1.)],[])
    elif change == 'stale':
        n._book_contact_force_samples[MODEL] = ([ForceSample(now.nanoseconds-200_000_000,4.)],)*2
    elif change == 'arm_velocity': n._joint_velocities['arm_left_1_joint'] = .0011
    elif change == 'missing_arm_velocity': del n._joint_velocities['arm_left_1_joint']
    elif change == 'right_context': n.joints['arm_right_1_joint'] = .0011
    elif change == 'head_reference': gate.reference['joints']['head_1_joint'] = -.0011
    elif change == 'base': n._staging_odom['pose'] = (2.003,-.1,.02)
    elif change == 'finger': n.joints['gripper_left_finger_joint'] += .00002
    elif change == 'hazard': n._payload_hazard_latched = 'contact_lost'
    elif change == 'baseline': gate.baseline_effort = float('nan')
    elif change == 'pending': n._pending_retained_acceptances.add(object())
    with pytest.raises(module.LiftPressureRejected) as exc:
        gate.send(send)
    if reason not in ('weak', 'stale'):
        assert reason in str(exc.value)
    assert sent == []


@pytest.mark.parametrize('fault', ['weak', 'force', 'cancel', 'identity', 'velocity', 'context'])
def test_callback_during_clock_catchup_invalidates_dispatch(monkeypatch, fault):
    gate, n, now, wall, sent, _, make_joint, send = fixture(monkeypatch)
    def update():
        if fault in ('weak','force'):
            n._on_contacts(force_message(now.nanoseconds, (RIGHT,BOOK,.1 if fault=='weak' else 21.)))
        elif fault=='cancel': n._cancel.set()
        elif fault=='identity': n._on_contacts(force_message(now.nanoseconds,(RIGHT,'book_col_2_row_1_red::book_base_link::collision',4.)))
        elif fault=='velocity':
            msg=make_joint(); msg.velocity[msg.name.index('arm_left_1_joint')]=.002
            n._on_joint_state(msg)
        elif fault=='context': n.joints['head_1_joint']=.002
    wall.hook = update
    with pytest.raises(module.LiftPressureRejected): gate.send(send)
    assert sent == []


def test_fixed_future_snapshot_catches_up_without_chasing_new_producer_frames(monkeypatch):
    gate, n, now, wall, sent, _, make_joint, send = fixture(monkeypatch)
    stamp = now.nanoseconds+2_000_000
    n._on_contacts(force_message(stamp,(LEFT,BOOK,4.),(RIGHT,BOOK,4.2)))
    def newer():
        n._on_contacts(force_message(now.nanoseconds+2_000_000,(LEFT,BOOK,4.),(RIGHT,BOOK,4.2)))
        n._on_joint_state(make_joint(now.nanoseconds+2_000_000))
    wall.hook = newer
    assert gate.send(send) == 'accepted-future'
    assert len(sent)==1 and wall.seconds < 100.01


@pytest.mark.parametrize('other', [
    'book_col_2_row_1_red::book_base_link::collision',
    'book_col_3_row_2_blue::book_base_link::collision',
    'prefixbook_col_3_row_2_redsuffix::book_base_link::collision',
    'base_link_book_collision', 'shelf::shelf_link::collision',
])
def test_filtered_raw_contact_fault_latches_through_later_good_frame(monkeypatch,other):
    gate,n,now,wall,sent,_,_,send=fixture(monkeypatch)
    def bad_then_good():
        n._on_contacts(force_message(now.nanoseconds,(RIGHT,other,4.)))
        n._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,4.),(RIGHT,BOOK,4.2)))
    wall.hook=bad_then_good
    with pytest.raises(module.LiftPressureRejected,match='unexpected_held_contact_identity'):
        gate.send(send)
    assert not sent


@pytest.mark.parametrize('kind',['nan','missing_wrench','palm'])
def test_malformed_or_unintended_held_hand_contact_cannot_hide_behind_good_history(monkeypatch,kind):
    gate,n,now,wall,sent,_,_,send=fixture(monkeypatch)
    def bad():
        hand='tiago_pro::gripper_left_base_link::collision' if kind=='palm' else RIGHT
        msg=force_message(now.nanoseconds,(hand,BOOK,float('nan') if kind=='nan' else 4.))
        if kind=='missing_wrench':msg.contacts[0].wrenches=[]
        n._on_contacts(msg)
    wall.hook=bad
    with pytest.raises(module.LiftPressureRejected,match='held_contact_force|held_hand_contact'):gate.send(send)
    assert not sent


def test_backfilled_weak_sample_inside_prior_proof_invalidates_admission(monkeypatch):
    gate,n,now,wall,sent,_,_,send=fixture(monkeypatch)
    old_clock=now.nanoseconds
    def backfill():
        # A newly delivered distinct older producer frame breaks the 50 ms
        # qualifying suffix even though both latest samples remain strong.
        n._on_contacts(force_message(old_clock-21_000_000,(LEFT,BOOK,.1)))
    wall.hook=backfill
    with pytest.raises(module.LiftPressureRejected): gate.send(send)
    assert not sent


def test_new_passive_feedback_after_geometry_is_not_silently_ignored(monkeypatch):
    gate,n,_,wall,sent,_,_,send=fixture(monkeypatch)
    wall.hook=lambda:n.joints.update(gripper_left_outer_knuckle_left_joint=.1)
    with pytest.raises(module.LiftPressureRejected,match='availability_changed'): gate.send(send)
    assert not sent


@pytest.mark.parametrize('invalid',['nan','stale'])
def test_partial_reported_passive_stays_finite_and_fresh_during_gate(monkeypatch,invalid):
    gate,n,now,wall,sent,_,_,send=fixture(monkeypatch)
    name='gripper_left_outer_knuckle_left_joint'
    gate.checked['reported_passive_joints']=[name]
    n.joints[name]=.1
    n._joint_stamps_ns[name]=now.nanoseconds
    def invalidate():
        if invalid=='nan':n.joints[name]=float('nan')
        else:n._joint_stamps_ns[name]=now.nanoseconds-400_000_000
    wall.hook=invalidate
    with pytest.raises(module.LiftPressureRejected,match='reported_passive_feedback_invalid'):gate.send(send)
    assert not sent


def test_dual_context_drift_can_fail_despite_each_side_within_original_tolerance(monkeypatch):
    gate,n,_,wall,sent,_,_,send=fixture(monkeypatch)
    n.joints['arm_right_1_joint']=-.0008
    gate.checked['geometry_context']['right_positions']=(-.0008,)+(0.,)*6
    wall.hook=lambda:n.joints.update(arm_right_1_joint=.0008)
    with pytest.raises(module.LiftPressureRejected,match='collision_context_changed'): gate.send(send)
    assert not sent


def test_clock_stall_rejects_and_clock_wait_does_not_hold_sensor_lock(monkeypatch):
    gate, n, now, wall, sent, _, _, send = fixture(monkeypatch)
    def stalled(seconds):
        assert not n._lock.locked()
        wall.seconds += seconds
    monkeypatch.setattr(module.time,'sleep',stalled)
    with pytest.raises(module.LiftPressureRejected, match='clock_catchup_timeout'): gate.send(send)
    assert sent==[] and wall.seconds < 100.51


def test_clock_reversal_between_pressure_attempts_cannot_reset_monotonic_guard(monkeypatch):
    gate,n,now,wall,sent,_,_,send=fixture(monkeypatch)
    n._book_contact_force_samples[MODEL][0].append(ForceSample(now.nanoseconds+1,.24))
    calls=0
    def reversing(seconds):
        nonlocal calls
        calls+=1
        wall.seconds+=seconds
        now.nanoseconds+=2_000_000 if calls==1 else -1_000_000
    monkeypatch.setattr(module.time,'sleep',reversing)
    with pytest.raises(module.LiftPressureRejected,match='clock_reversed'):gate.send(send)
    assert not sent and calls==2


def test_transport_exception_is_not_misreported_as_prepublication_rejection(monkeypatch):
    gate, _, _, _, _, _, _, _ = fixture(monkeypatch)
    def send(): raise OSError('unknown acceptance')
    with pytest.raises(OSError, match='unknown acceptance'): gate.send(send)
    assert gate.send_started


def test_logging_failure_cannot_hide_already_sent_acceptance_future(monkeypatch):
    gate,n,_,_,sent,_,_,send=fixture(monkeypatch)
    def fail(*args,**kwargs): raise OSError('log failure')
    n._publish_status=fail
    assert gate.send(send)=='accepted-future'
    assert len(sent)==1


@pytest.mark.parametrize('eventual_pressure', [True, False])
def test_bounded_same_grip_wait_requires_new_qualifying_window(monkeypatch,eventual_pressure):
    gate,n,now,wall,sent,events,make_joint,send=fixture(monkeypatch)
    start=now.nanoseconds
    for side in n._book_contact_force_samples[MODEL]:
        for index,sample in enumerate(side):
            side[index]=replace(sample,force_newtons=.24)
    def refresh():
        force=4. if eventual_pressure and now.nanoseconds-start>=80_000_000 else .24
        n._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,force),(RIGHT,BOOK,force)))
        n._on_joint_state(make_joint())
        n._staging_odom['stamp_ns']=now.nanoseconds
    wall.hook=refresh
    if eventual_pressure:
        assert gate.send(send)=='accepted-future'
        assert 130_000_000 <= sent[0]-start < 160_000_000
    else:
        with pytest.raises(module.LiftPressureRejected,match='timeout|deadline'): gate.send(send)
        assert not sent and 105. <= wall.seconds < 105.01
    assert len([event for event,_ in events if event=='lift_first_pressure_waiting'])==1
    # The helper has no gripper publisher or command API; it only receives the
    # eventual arm callback. The held commanded/observed aperture is unchanged.
    assert n.joints['gripper_left_finger_joint']==.0183


@pytest.mark.parametrize('fault', ['force', 'missing', 'velocity', 'cancel'])
def test_hard_fault_during_weak_wait_stops_without_using_remaining_budget(monkeypatch,fault):
    gate,n,now,wall,sent,_,make_joint,send=fixture(monkeypatch)
    start=now.nanoseconds
    n._book_contact_force_samples[MODEL][0].append(ForceSample(start+1,.24))
    def refresh():
        n._on_joint_state(make_joint())
        n._staging_odom['stamp_ns']=now.nanoseconds
        n._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,.24),(RIGHT,BOOK,.24)))
        if now.nanoseconds-start>=20_000_000:
            if fault=='force': n._on_contacts(force_message(now.nanoseconds,(RIGHT,BOOK,21.)))
            elif fault=='missing': n._book_contact_force_samples[MODEL][0].clear()
            elif fault=='velocity': n._joint_velocities['arm_left_1_joint']=.002
            else: n._cancel.set()
    wall.hook=refresh
    with pytest.raises(module.LiftPressureRejected):gate.send(send)
    assert not sent and wall.seconds<100.1


def test_sender_rejection_never_registers_pending_or_enters_recovery(monkeypatch):
    gate, n, _, _, sent, _, _, _ = fixture(monkeypatch)
    n._payload_hazard_reason = lambda **kw: None
    n.arm_client = SimpleNamespace(send_goal_async=lambda goal: sent.append(goal))
    n._gripper_feedback_samples.append(replace(n._gripper_feedback_samples[-1],velocity=1.))
    with pytest.raises(module.LiftPressureRejected):
        n._send_retained_arm_trajectory(object(),1.,[(None,1.,'initial_shelf_lift')], 'pick',initial_pressure_gate=gate)
    assert not n._pending_retained_acceptances and not n._goal_handles and not sent
    assert not n._cancel.is_set()


def test_missing_velocity_array_is_recorded_invalid_in_real_joint_callback(monkeypatch):
    gate, n, _, _, sent, _, make_joint, send = fixture(monkeypatch)
    msg=make_joint(); msg.velocity=[]
    n._on_joint_state(msg)
    with pytest.raises(module.LiftPressureRejected): gate.send(send)
    assert not sent


@pytest.mark.parametrize('separate_messages',[False,True])
@pytest.mark.parametrize('maximum', [20., 30.])
def test_aggregate_held_force_fault_survives_probe_reset(monkeypatch,separate_messages,maximum):
    gate,n,now,_,sent,_,_,send=fixture(monkeypatch)
    n.adaptive_contact_force_maximum = maximum
    inner='tiago_pro::gripper_left_inner_finger_left_link::collision'
    each = maximum/2 + 1.
    pairs=((LEFT,BOOK,each),(inner,BOOK,each))
    assert not n._adaptive_close_active
    if separate_messages:
        for pair in pairs:n._on_contacts(force_message(now.nanoseconds,pair))
    else:n._on_contacts(force_message(now.nanoseconds,*pairs))
    assert n._book_contact_force_samples[MODEL][0][-1].force_newtons==maximum+2.
    assert n._held_grip_sensor_fault=='force_overload'
    # The normal fresh-retention probe clears histories, not a held fault.
    n._clear_target_contact_samples()
    for stamp in range(now.nanoseconds-60_000_000,now.nanoseconds+1,2_000_000):
        n._on_contacts(force_message(stamp,(LEFT,BOOK,4.),(RIGHT,BOOK,4.2)))
    assert max(sample.force_newtons for sample in n._book_contact_force_samples[MODEL][0])==4.
    with pytest.raises(module.LiftPressureRejected,match='force_overload'):gate.send(send)
    assert not sent


@pytest.mark.parametrize('force,allowed', [(20.3735, True), (30., True), (30.0001, False)])
def test_explicit_shared_thirty_newton_cap_also_applies_at_fresh_lift(monkeypatch, force, allowed):
    gate,n,now,_,sent,_,_,send=fixture(monkeypatch)
    n.adaptive_contact_force_maximum = 30.
    assert gate.minimum_force == 3.5
    n._on_contacts(force_message(now.nanoseconds, (RIGHT,BOOK,force)))
    if allowed:
        assert gate.send(send) == 'accepted-future'
        assert len(sent) == 1
    else:
        assert n._held_grip_sensor_fault == 'force_overload'
        n._on_contacts(force_message(now.nanoseconds+2_000_000, (RIGHT,BOOK,4.2)))
        with pytest.raises(module.LiftPressureRejected, match='force_overload'):
            gate.send(send)
        assert not sent


def test_duplicate_held_pair_does_not_create_aggregate_overload(monkeypatch):
    gate,n,now,_,sent,_,_,send=fixture(monkeypatch)
    message=force_message(now.nanoseconds,(LEFT,BOOK,11.))
    n._on_contacts(message)
    n._on_contacts(message)
    assert n._book_contact_force_samples[MODEL][0][-1].force_newtons==11.
    assert getattr(n,'_held_grip_sensor_fault',None) is None
    assert gate.send(send)=='accepted-future'
    assert len(sent)==1
