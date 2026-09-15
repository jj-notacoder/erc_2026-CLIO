"""Production recovery stops retained lower-shelf payloads before actuation."""
import threading
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip('rclpy')
from erc_phase1_solution import manipulation_node as manipulation
from erc_phase1_solution.motion_profiles import HOME
from test_shutdown import _mocked_pick_trace


def recovery_node():
    node = object.__new__(manipulation.ManipulationNode)
    node._lock = threading.Lock()
    node._target_robot_contact_latched = False
    node._gravity_supported_payload = False
    node._publish_status = Mock()
    node._target_contact_recent = Mock(return_value=True)
    node._fresh_retention_probe = Mock(return_value=True)
    node._retention_after_leg = Mock(return_value=True)
    node._move_arm_solution = Mock(return_value=True)
    node._move_torso = Mock(return_value=True)
    node._open_gripper = Mock(return_value=True)
    return node


def recover(node, **options):
    return node._recover_closed_pick(
        cause=options.pop('cause', 'invalid_stock_retained_feedback'),
        remaining_carried_legs=[(HOME.copy(), 1., 'extraction')],
        unloaded_recovery_route=[HOME.copy()], **options)


def test_lower_generic_failure_holds_before_arm_torso_or_jaw_commands():
    node = recovery_node()
    with pytest.raises(RuntimeError, match='pick_recovery_failed'):
        recover(node, lower_shelf_pick=True)
    node._move_arm_solution.assert_not_called()
    node._move_torso.assert_not_called()
    node._open_gripper.assert_not_called()
    node._retention_after_leg.assert_not_called()
    node._publish_status.assert_called_once_with(
        'carried_recovery', command='pick', cause='invalid_stock_retained_feedback',
        reason='lower_shelf_recovery_certificate_missing',
        retained_before_recovery=True, gripper_opened=False,
        recovery_succeeded=False, recovery_halted=True, retained_stop=True)


@pytest.mark.parametrize('cause,exception,opened,retained', [
    ('motion_failed', 'pick_recovery_failed', False, True),
    ('torso_motion_failed', 'pick_recovery_failed', False, True),
    ('final_grasp_settle_failed', 'pick_recovery_failed', False, True),
    ('final_grasp_width_invalid', 'pick_recovery_failed', False, True),
    ('contact_lost', 'pick_payload_lost', True, False),
    ('payload_robot_contact', 'pick_payload_lost', False, False),
])
def test_known_failures_keep_their_specific_retention_behavior(
        cause, exception, opened, retained):
    node = recovery_node()
    with pytest.raises(RuntimeError, match=exception):
        recover(node, cause=cause, lower_shelf_pick=True)
    node._move_arm_solution.assert_not_called()
    node._move_torso.assert_not_called()
    assert node._open_gripper.call_count == int(opened)
    expected = dict(command='pick', cause=cause,
        retained_before_recovery=retained, gripper_opened=opened,
        recovery_succeeded=False, recovery_halted=True)
    if cause != 'contact_lost':
        expected['retained_stop'] = True
    node._publish_status.assert_called_once_with('carried_recovery', **expected)


def test_top_default_and_explicit_false_complete_the_existing_checked_recovery():
    for options in ({}, {'lower_shelf_pick': False}):
        node = recovery_node()
        order = []
        node._move_arm_solution.side_effect = lambda *args: order.append('arm') or True
        node._move_torso.side_effect = lambda *args: order.append('torso') or True
        node._open_gripper.side_effect = lambda: order.append('open') or True
        assert recover(node, **options) is False
        assert order == ['arm', 'torso', 'open', 'arm']
        assert [call.args[1] for call in node._move_arm_solution.call_args_list] == [1., 2.2]
        assert all(np.array_equal(call.args[0], HOME)
                   for call in node._move_arm_solution.call_args_list)
        node._move_torso.assert_called_once_with(float(HOME[0]), 2.)
        node._retention_after_leg.assert_called_once_with('pick', 'recovery_extraction', 0)
        node._publish_status.assert_called_once_with(
            'carried_recovery', command='pick', cause='invalid_stock_retained_feedback',
            retained_before_recovery=True, gripper_opened=True, recovery_succeeded=True)


@pytest.mark.parametrize('height,lower_shelf', [(1.25, True), (1.58, False)])
def test_actual_pick_passes_row_context_when_final_contact_is_lost(height, lower_shelf):
    result = _mocked_pick_trace([.67, -.05, height], final_contact=False)
    assert not result['succeeded']
    assert result['closed_recoveries']
    assert all(recovery['lower_shelf_pick'] is lower_shelf
               for recovery in result['closed_recoveries'])
