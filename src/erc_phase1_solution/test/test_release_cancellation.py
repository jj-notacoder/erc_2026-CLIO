"""Exercise actual ROS-message publication and cancellation admission, without spinning."""
from types import SimpleNamespace
import json
import threading

from std_msgs.msg import String

from erc_phase1_solution import manipulation_node


def release_node(monkeypatch):
    node = object.__new__(manipulation_node.ManipulationNode)
    node._lock = threading.RLock()
    node._adaptive_command_lock = threading.RLock()
    node._cancel = threading.Event()
    node._goal_handles = []
    node.stock_gripper_close_diagnostic_enabled = True
    node._held_book_corners = object()
    node._payload_monitor_enabled = True
    node._retention_probe_active = False
    node._gripper_open_confirmed = False
    node._adaptive_hold_sent = False
    node._adaptive_motion_halt_reason = None
    node.gripper_open = .069
    node.gripper_settle = 1.2
    node.preclose_aperture_geometry_enabled = False
    commands, cleared, slept = [], [], []
    node.gripper_pub = SimpleNamespace(publish=lambda msg: commands.append(
        float(msg.points[0].positions[0])))
    node._publish_status = lambda *args, **kwargs: None
    node._adaptive_motion_feedback = lambda: (SimpleNamespace(position=.017), None)
    node._clear_target_contact_samples_unlocked = lambda **kwargs: cleared.append(kwargs)

    def wait(_):
        assert not node._adaptive_command_lock._is_owned()
        return not node._cancel.is_set()

    def sleep(duration):
        assert not node._adaptive_command_lock._is_owned()
        slept.append(duration)

    node._wait_sim_duration = wait
    monkeypatch.setattr(manipulation_node.time, 'sleep', sleep)
    return node, commands, cleared, slept, sleep


def test_release_cancelled_before_publication_preserves_payload(monkeypatch):
    node, commands, cleared, _, _ = release_node(monkeypatch)
    held = node._held_book_corners
    node._cancel.set()
    measured = []
    assert not node._open_gripper(verify_measurement=lambda: measured.append(True))
    assert commands == [] and measured == [] and cleared == []
    assert node._held_book_corners is held
    assert node._payload_monitor_enabled


def test_release_cancel_between_duplicates_leaves_measured_hold_last(monkeypatch):
    node, commands, cleared, slept, original_sleep = release_node(monkeypatch)
    held = node._held_book_corners
    measured = []

    def sleep(duration):
        original_sleep(duration)
        if len(slept) == 1:
            node._on_command(String(data=json.dumps({'event': 'cancel'})))

    monkeypatch.setattr(manipulation_node.time, 'sleep', sleep)
    assert not node._open_gripper(verify_measurement=lambda: measured.append(True))
    assert commands == [.069, .017]
    assert measured == [] and cleared == []
    assert node._held_book_corners is held
    assert node._payload_monitor_enabled


def test_release_clears_payload_only_after_successful_measurement(monkeypatch):
    node, commands, cleared, _, _ = release_node(monkeypatch)
    held = node._held_book_corners

    def verify():
        assert node._held_book_corners is held
        assert commands == [.069, .069] and cleared == []
        return True

    assert node._open_gripper(verify_measurement=verify)
    assert node._held_book_corners is None
    assert node._gripper_open_confirmed
    assert len(cleared) == 1


def test_release_unverified_measurement_preserves_state_without_reopening(monkeypatch):
    node, commands, cleared, _, _ = release_node(monkeypatch)
    held = node._held_book_corners
    assert not node._open_gripper(verify_measurement=lambda: False)
    assert commands == [.069, .069] and cleared == []
    assert node._held_book_corners is held
    assert node._payload_monitor_enabled and not node._gripper_open_confirmed
