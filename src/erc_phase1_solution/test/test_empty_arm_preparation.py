"""Frame, evidence and recovery boundaries for empty-arm preparation."""

import math
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from erc_phase1_solution.empty_arm_preparation import (
    check_empty_stationary, point_in_pose, validate_preparation_request,
)
from erc_phase1_solution.motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS


@pytest.mark.parametrize('yaw', [-2.9, -1.5, 0., .8, 2.4])
@pytest.mark.parametrize('lateral', [-.10, -.05, .03])
def test_prediction_uses_actual_rotated_base_and_book_lateral_offset(yaw, lateral):
    start = np.array([2.7, -1.3, yaw])
    heading = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-heading[1], heading[0]])
    goal = start.copy()
    goal[:2] += .2 * heading
    xy = goal[:2] + .79 * heading + lateral * left
    payload = dict(plan_id='trial-stage1', book_odom=[*xy, 1.58],
                   book_stamp_ns=1_000_000_000, staging_pose=start.tolist(),
                   final_goal=goal.tolist())
    *_, front, advance = validate_preparation_request(
        payload, 1_200_000_000, start,
    )
    np.testing.assert_allclose(front, [.79, lateral, 1.58], atol=1e-12)
    assert advance == pytest.approx(.20)
    np.testing.assert_allclose(point_in_pose(payload['book_odom'], start),
                               [.99, lateral, 1.58], atol=1e-12)


def request():
    return dict(plan_id='stage1', book_odom=[.99, -.05, 1.58],
                book_stamp_ns=1_000_000_000, staging_pose=[0, 0, 0],
                final_goal=[.2, 0, 0])


@pytest.mark.parametrize('mutation,reason', [
    ({'plan_id': ''}, 'plan_id'),
    ({'book_stamp_ns': -10_000_000_000}, 'stale'),
    ({'final_goal': [.2, .03, 0]}, 'straight'),
    ({'final_goal': [.2, 0, .05]}, 'straight'),
    ({'final_goal': [-.2, 0, 0]}, 'straight'),
    ({'book_odom': [float('nan'), 0, 1.58]}, 'book_odom'),
    ({'book_odom': [.99, -.05, 1.1]}, 'top-row'),
])
def test_bad_or_stale_preparation_requests_are_rejected(mutation, reason):
    payload = request()
    payload.update(mutation)
    with pytest.raises(ValueError, match=reason):
        validate_preparation_request(payload, 1_200_000_000, [0, 0, 0])


def test_base_shift_during_preparation_is_rejected():
    with pytest.raises(ValueError, match='pose changed'):
        validate_preparation_request(request(), 1_200_000_000, [.045, 0, 0])


def measured_node():
    names = (*IK_JOINTS, *RIGHT_ARM_JOINTS, 'head_1_joint', 'head_2_joint',
             'gripper_left_finger_joint')
    return SimpleNamespace(
        _lock=threading.Lock(), joints={name: 0. for name in names},
        _joint_stamps_ns={name: 1_000_000_000 for name in names},
        _staging_odom=dict(stamp_ns=1_000_000_000, pose=[0, 0, 0],
                           linear_speed=0., angular_speed=0.),
        _empty_hand_contact=(1_000_000_000, False),
        count_publishers=lambda topic: 30,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(
            nanoseconds=1_100_000_000)),
    )


def test_recent_stationary_empty_hand_measurements_are_required():
    node = measured_node()
    assert check_empty_stationary(node, [0, 0, 0])[0] == node.joints
    node._held_book_corners = np.zeros((8, 3))
    with pytest.raises(RuntimeError, match='payload'):
        check_empty_stationary(node)


@pytest.mark.parametrize('change,reason', [
    (lambda n: n._joint_stamps_ns.update(torso_lift_joint=1), 'joint measurement stale'),
    (lambda n: n._staging_odom.update(stamp_ns=1), 'odometry is stale'),
    (lambda n: n._staging_odom.update(linear_speed=.02), 'base is moving'),
    (lambda n: setattr(n, 'count_publishers', lambda topic: 0), 'publisher is unavailable'),
    (lambda n: setattr(n, '_empty_hand_contact', (1_000_000_000, True)), 'external contact'),
])
def test_unknown_moving_or_contacting_state_never_qualifies_as_empty(change, reason):
    node = measured_node()
    change(node)
    with pytest.raises(RuntimeError, match=reason):
        check_empty_stationary(node)


def test_event_only_contact_topic_does_not_require_an_empty_heartbeat():
    node = measured_node()
    node._empty_hand_contact = None
    assert check_empty_stationary(node)[0] == node.joints
    # A non-gripper callback must not erase an external hand contact.
    node._last_external_hand_contact_ns = 1_000_000_000
    node._empty_hand_contact = (1_050_000_000, False)
    with pytest.raises(RuntimeError, match='external contact'):
        check_empty_stationary(node)


def test_staged_recovery_holds_instead_of_using_generic_home_motion():
    pytest.importorskip('rclpy')
    from erc_phase1_solution.manipulation_node import ManipulationNode
    node = object.__new__(ManipulationNode)
    node._empty_arm_staged = True
    node._follow = lambda *a, **kw: pytest.fail('unchecked HOME command')
    assert node._return_arm_home() is False
    with pytest.raises(RuntimeError, match='checked recovery'):
        node._stow()


def mission_prepared():
    pytest.importorskip('rclpy')
    from erc_phase1_solution.mission_manager import MissionManager
    import time
    node = object.__new__(MissionManager)
    node.state = 'PREPARE_EMPTY_ARM'
    node.finished = False
    node.wall_started = time.monotonic()
    node.trial_timeout = 1200.
    node._empty_stage_plan_id = 'current-stage'
    node.manip_event = dict(event='succeeded', command='prepare_pick_approach',
                            plan_id='current-stage', prepared_stamp_ns=1_000_000_000,
                            final_goal=[1., 2., .3], prepared_joints={'arm_left_1_joint': .5},
                            advance_bounds=[[-.4, -.3, 0], [.53, .3, 1.7]])
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(
        nanoseconds=1_100_000_000))
    node.calls = []
    node._navigate = lambda *args, **kw: node.calls.append(('navigate', args, kw))
    node._abort = lambda reason: node.calls.append(('abort', reason))
    node._set_state = lambda state: node.calls.append(('state', state))
    return node


@pytest.mark.parametrize('changes', [
    {'plan_id': 'previous-stage'}, {'prepared_stamp_ns': 1}, {'prepared_joints': {}},
])
def test_mission_rejects_stale_or_mismatched_preparation_before_base_motion(changes):
    node = mission_prepared()
    node.manip_event.update(changes)
    node._tick()
    assert node.calls == [('abort', 'empty_arm_preparation_certificate_invalid')]


def test_mission_binds_prepared_configuration_to_one_straight_advance():
    node = mission_prepared()
    node._tick()
    _, args, fields = node.calls[0]
    assert args == (1., 2., .3, 'empty_arm_final_advance')
    assert fields['profile'] == 'empty_arm_advance'
    assert fields['plan_id'] == 'current-stage'
    assert fields['prepared_joints'] == node.manip_event['prepared_joints']
    assert fields['advance_bounds'] == node.manip_event['advance_bounds']
    assert node.calls[1] == ('state', 'ALIGN_BOOK')
    assert node._empty_stage_pending is False


def test_transient_contact_during_preparation_latches_and_cancels_motion():
    pytest.importorskip('rclpy')
    from erc_phase1_solution.manipulation_node import ManipulationNode
    node = object.__new__(ManipulationNode)
    node._lock = threading.Lock()
    node._cancel = threading.Event()
    node._empty_arm_motion_active = True
    node.target_colour = 'red'
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(
        nanoseconds=1_000_000_000))
    node._publish_status = lambda *a, **kw: None
    touch = SimpleNamespace(contacts=[SimpleNamespace(
        collision1=SimpleNamespace(name='robot::gripper_left_fingertip_left_link::collision'),
        collision2=SimpleNamespace(name='bookshelf::shelf::collision'),
    )])
    node._on_contacts(touch)
    node._on_contacts(SimpleNamespace(contacts=[]))
    assert node._empty_hand_contact == (1_000_000_000, False)
    assert node._empty_arm_contact_latched is True
    assert node._cancel.is_set()


def test_empty_hand_contact_hazard_aborts_mission_during_base_advance():
    node = mission_prepared()
    from erc_phase1_solution.mission_manager import String, encode_event
    node.state = 'ALIGN_BOOK'
    node._log = lambda *a, **kw: None
    node._on_manipulation_status(String(data=encode_event(
        'empty_arm_hazard', reason='external_hand_contact')))
    assert node.calls == [('abort', 'empty_arm_hazard:external_hand_contact')]
