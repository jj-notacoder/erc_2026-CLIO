"""Bounded visibility recovery never substitutes a remembered placement point."""

from dataclasses import replace
import math
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from erc_phase1_solution.bin_visibility_recovery import (
    BinApproachReference,
    verify_reacquired_bin,
    visibility_retreat_goal,
)
from erc_phase1_solution.mission_manager import MissionManager, PointStamped, String


REFERENCE = BinApproachReference((1., 0., .75), 10_000_000_000, (.28, 0., 0.))
NOW = 30_000_000_000


@pytest.mark.parametrize('yaw', [0., .6, -2.3, math.pi])
def test_retreat_is_short_straight_outward_in_any_heading(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    reference = BinApproachReference((c, s, .75), REFERENCE.stamp_ns, (.28*c, .28*s, yaw))
    goal = visibility_retreat_goal(reference, reference.goal, NOW, NOW, 0)
    assert math.dist(goal[:2], reference.goal[:2]) == pytest.approx(.12)
    assert goal[2] == yaw
    assert math.dist(goal[:2], reference.point[:2]) == pytest.approx(.84)


@pytest.mark.parametrize('reference,pose,stamp,now,attempts,reason', [
    (REFERENCE, REFERENCE.goal, NOW, NOW, 1, 'exhausted'),
    (None, REFERENCE.goal, NOW, NOW, 0, 'reference_expired'),
    (replace(REFERENCE, stamp_ns=-1), REFERENCE.goal, NOW, NOW, 0, 'reference_expired'),
    (REFERENCE, REFERENCE.goal, NOW, 101_000_000_000, 0, 'reference_expired'),
    (REFERENCE, REFERENCE.goal, NOW+1, NOW, 0, 'pose_stale'),
    (REFERENCE, REFERENCE.goal, NOW-500_000_001, NOW, 0, 'pose_stale'),
    (REFERENCE, (.5, 0., 0.), NOW, NOW, 0, 'displaced'),
    (REFERENCE, (.28, 0., .13), NOW, NOW, 0, 'heading_changed'),
    (REFERENCE, (.28, float('nan'), 0.), NOW, NOW, 0, 'geometry'),
    (replace(REFERENCE, point=(.3, 0., .75)), REFERENCE.goal, NOW, NOW, 0, 'distance_invalid'),
    (replace(REFERENCE, point=(2., 0., .75)), REFERENCE.goal, NOW, NOW, 0, 'distance_invalid'),
    (replace(REFERENCE, point=(-.5, 0., .75)), REFERENCE.goal, NOW, NOW, 0, 'not_facing_bin'),
])
def test_retreat_refuses_unbounded_or_untrusted_geometry(reference, pose, stamp, now, attempts, reason):
    with pytest.raises(ValueError, match=reason):
        visibility_retreat_goal(reference, pose, stamp, now, attempts)


@pytest.mark.parametrize('point,stamp,after,reason', [
    (REFERENCE.point, NOW, NOW, 'not_new'),
    (REFERENCE.point, NOW-500_000_001, NOW-1_000_000_000, 'not_new'),
    (REFERENCE.point, NOW+1, NOW-1, 'not_new'),
    ((1.3, 0., .75), NOW, NOW-1, 'location_changed'),
    ((1., 0., float('inf')), NOW, NOW-1, 'geometry'),
])
def test_reacquisition_requires_new_fresh_same_location(point, stamp, after, reason):
    with pytest.raises(ValueError, match=reason):
        verify_reacquired_bin(REFERENCE, point, stamp, NOW, after)


def test_reacquisition_accepts_bounded_surface_observation_change():
    verify_reacquired_bin(REFERENCE, (1.03, .01, .76), NOW, NOW, NOW-1)


@pytest.fixture
def mission():
    node = object.__new__(MissionManager)
    node.finished = False
    node.wall_started = time.monotonic()
    node.trial_timeout = 1000.
    node.state = 'REACQUIRE_BIN'
    node.bin_visibility_recovery_enabled = True
    node.bin_point = node.bin_candidate = None
    node.bin_verified_ns = node.bin_invalidated_ns = -1
    node._bin_approach_reference = REFERENCE
    node._bin_visibility_attempts = 0
    node._bin_visibility_started_ns = None
    node._bin_reacquire_after_ns = NOW-1_000_000_000
    node.perception_timeout = 18.
    node.navigation_timeout = 50.
    node.manipulation_timeout = 180.
    node.nav_event = {'event': 'reached'}
    node.manip_event = None
    now = [NOW]
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=now[0]))
    node._elapsed_state = lambda: 18.1
    node._target_identity_confirmed = lambda: True
    node._log = Mock()
    node._perception_mode = Mock()
    node._abort = Mock()
    node._manipulate = Mock()
    transitions, navigation = [], []
    def set_state(state, **fields):
        node.state = state
        transitions.append(state)
    def navigate(*args, **kwargs):
        node.nav_event = None
        navigation.append((args, kwargs))
    node._set_state = set_state
    node._navigate = navigate
    node.tf_buffer = SimpleNamespace(
        lookup_transform=Mock(return_value=SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=30, nanosec=0)),
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=.28, y=0., z=0.),
                rotation=SimpleNamespace(x=0., y=0., z=0., w=1.),
            ),
        )),
        transform=Mock(return_value=SimpleNamespace(point=SimpleNamespace(x=1., y=0., z=.75))),
    )
    return node, now, transitions, navigation


def fresh_bin(node, stamp_ns):
    point = PointStamped()
    point.header.frame_id = 'depth_optical_frame'
    point.header.stamp.sec, point.header.stamp.nanosec = divmod(stamp_ns, 10**9)
    point.point.z = .84
    node.bin_point = point
    node.bin_verified_ns = stamp_ns
    return point


def test_timeout_retreats_once_with_carried_profile_then_reacquires_before_place(mission):
    node, now, transitions, navigation = mission
    node._tick()
    assert transitions == ['BIN_VISIBILITY_RETREAT']
    assert navigation[0][0][:3] == pytest.approx((.16, 0., 0.))
    assert navigation[0][0][3] == 'bin_visibility_retreat'
    assert navigation[0][1] == {'profile': 'carried_retreat'}
    node._manipulate.assert_not_called()
    assert node._bin_visibility_attempts == 1

    node.nav_event = {'event': 'reached'}
    now[0] += 3_000_000_000
    node._tick()
    assert node.state == 'HEAD_REACQUIRE_BIN'
    node._manipulate.assert_called_once_with('look_bin')
    node.manip_event = {'event': 'succeeded', 'command': 'look_bin'}
    node._tick()
    assert node.state == 'REACQUIRE_BIN'
    assert node._bin_reacquire_after_ns == now[0]
    node._elapsed_state = lambda: .1
    fresh_bin(node, now[0])  # A replay from before the head-finished boundary.
    node._tick()
    assert node.state == 'REACQUIRE_BIN'
    now[0] += 200_000_000
    fresh_bin(node, now[0])
    node._tick()
    assert node.state == 'PLACE'
    node._manipulate.assert_called_with('place')
    node._abort.assert_not_called()
    assert len(navigation) == 1


@pytest.mark.parametrize('enabled,attempts', [(False, 0), (True, 1)])
def test_disabled_or_exhausted_recovery_does_not_move(mission, enabled, attempts):
    node, _, _, navigation = mission
    node.bin_visibility_recovery_enabled = enabled
    node._bin_visibility_attempts = attempts
    node._tick()
    assert navigation == []
    node._abort.assert_called_once_with('collection_bin_reacquisition_failed')


@pytest.mark.parametrize('identity,reached', [(False, True), (True, False)])
def test_recovery_requires_selected_payload_and_completed_approach(mission, identity, reached):
    node, _, _, navigation = mission
    node._target_identity_confirmed = lambda: identity
    node.nav_event = {'event': 'reached' if reached else 'failed'}
    node._tick()
    assert navigation == []
    node._abort.assert_called_once_with('collection_bin_reacquisition_failed')


def test_missing_fresh_pose_cannot_create_recovery_motion(mission):
    node, _, _, navigation = mission
    node.tf_buffer.lookup_transform.return_value.header.stamp.sec = 29
    node._tick()
    assert navigation == []
    node._abort.assert_called_once_with('collection_bin_reacquisition_failed')


@pytest.mark.parametrize('state', ['BIN_VISIBILITY_RETREAT', 'HEAD_REACQUIRE_BIN', 'REACQUIRE_BIN'])
@pytest.mark.parametrize('elapsed', [35_000_000_001, -1])
def test_total_recovery_deadline_cannot_reset_between_states(mission, state, elapsed):
    node, now, _, navigation = mission
    node.state = state
    node._bin_visibility_started_ns = now[0]-elapsed
    node._tick()
    assert navigation == []
    node._manipulate.assert_not_called()
    node._abort.assert_called_once_with('bin_visibility_recovery_timeout')


@pytest.mark.parametrize('verified_offset,invalidated_offset', [(-1, -1), (0, 0), (0, 1)])
def test_unverified_or_invalidated_point_never_authorizes_place(mission, verified_offset, invalidated_offset):
    node, now, _, _ = mission
    node._elapsed_state = lambda: .1
    fresh_bin(node, now[0])
    node.bin_verified_ns = now[0]+verified_offset
    node.bin_invalidated_ns = now[0]+invalidated_offset
    node._tick()
    node._manipulate.assert_not_called()
    assert node.state == 'REACQUIRE_BIN'


def test_different_verified_bin_location_aborts_without_place(mission):
    node, now, _, _ = mission
    fresh_bin(node, now[0])
    node.tf_buffer.transform.return_value.point.x = 1.4
    node._tick()
    node._abort.assert_called_once_with('bin_reacquisition_association_failed')
    node._manipulate.assert_not_called()


def test_retreat_navigation_failure_does_not_release_or_replan(mission):
    node, _, _, navigation = mission
    node.state = 'BIN_VISIBILITY_RETREAT'
    node.nav_event = {'event': 'failed', 'reason': 'obstacle'}
    node._tick()
    assert navigation == []
    node._manipulate.assert_not_called()
    node._abort.assert_called_once_with('bin_visibility_retreat_failed')


def test_payload_hazard_during_visibility_retreat_aborts(mission):
    node, _, _, navigation = mission
    node.state = 'BIN_VISIBILITY_RETREAT'
    node._on_manipulation_status(String(data='{"event":"payload_hazard","reason":"contact_lost"}'))
    node._abort.assert_called_once_with('payload_hazard:contact_lost')
    assert navigation == []
