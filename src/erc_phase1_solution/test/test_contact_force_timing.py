"""Producer-time and collision-pair force regressions using real ROS messages.

These callback tests do not create nodes, publish commands, or claim physical
retention. They exercise how independent sensor frames become pressure evidence.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
from ros_gz_interfaces.msg import Contact, Contacts, JointWrench

from erc_phase1_solution import manipulation_node


MODEL = 'book_col_3_row_2_red'
BOOK = f'{MODEL}::book_base_link::base_link_book_collision'
LEFT = 'tiago_pro::gripper_left_fingertip_left_link::collision'
LEFT_INNER = 'tiago_pro::gripper_left_inner_finger_left_link::collision'
RIGHT = 'tiago_pro::gripper_left_fingertip_right_link::collision'


def force_message(stamp_ns, *pairs):
    message = Contacts()
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp_ns, 10**9)
    for first, second, force in pairs:
        contact = Contact()
        contact.collision1.name = first
        contact.collision2.name = second
        wrench = JointWrench()
        # Keep body selection observable: the opposite body's value must not
        # contribute to the gripper force or the duplicate-pair identity.
        wrench.body_1_wrench.force.x = float(force if 'gripper_left_' in first else 100.)
        wrench.body_2_wrench.force.x = float(force if 'gripper_left_' in second else 100.)
        contact.wrenches = [wrench]
        message.contacts.append(contact)
    return message


def contact_node(clock_ns=10_000_000_000):
    node = object.__new__(manipulation_node.ManipulationNode)
    now = SimpleNamespace(nanoseconds=clock_ns)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    node.target_colour = 'red'
    node._target_book_model = MODEL
    node._book_contact_samples = {}
    node._book_contact_force_samples = {}
    node._left_target_contact_ns = node._right_target_contact_ns = 0
    node._contact_epoch = node._contact_generation = 0
    node._adaptive_close_active = True
    node._adaptive_overload_latched = None
    node.adaptive_contact_force_maximum = 8.
    node.grasp_contact_max_age = .75
    node.joints = {'gripper_left_finger_joint': .018}
    node._publish_status = lambda *args, **kwargs: None
    return node, now


def history(node, side=0):
    return tuple(node._book_contact_force_samples[MODEL][side])


def test_lagging_clock_preserves_distinct_producer_frames_without_force_inflation():
    node, now = contact_node()
    stamps = [now.nanoseconds + offset for offset in (10_000_000, 40_000_000, 70_000_000)]
    for stamp in stamps:
        node._on_contacts(force_message(stamp, (LEFT, BOOK, 3.), (RIGHT, BOOK, 3.)))

    for side in (0, 1):
        assert history(node, side) == tuple(
            manipulation_node.ForceSample(stamp, 3.) for stamp in stamps)
    assert node._left_target_contact_ns == node._right_target_contact_ns == stamps[-1]
    assert node._adaptive_overload_latched is None


@pytest.mark.parametrize('reverse_second', [False, True])
@pytest.mark.parametrize('second_force,expected', [(3., 3.), (2., 3.), (4., 4.)])
def test_repeated_pair_at_one_stamp_keeps_conservative_maximum_not_sum(
    reverse_second, second_force, expected,
):
    node, now = contact_node()
    node._on_contacts(force_message(now.nanoseconds, (LEFT, BOOK, 3.)))
    pair = (BOOK, LEFT, second_force) if reverse_second else (LEFT, BOOK, second_force)
    node._on_contacts(force_message(now.nanoseconds, pair))

    assert history(node) == (manipulation_node.ForceSample(now.nanoseconds, expected),)
    assert node._adaptive_overload_latched is None


@pytest.mark.parametrize('separate_messages', [False, True])
def test_distinct_same_side_pairs_at_one_stamp_add_without_faking_temporal_samples(
    separate_messages,
):
    node, now = contact_node()
    pairs = [(LEFT, BOOK, 3.), (LEFT_INNER, BOOK, 4.)]
    if separate_messages:
        for pair in pairs:
            node._on_contacts(force_message(now.nanoseconds, pair))
    else:
        node._on_contacts(force_message(now.nanoseconds, *pairs))
    node._on_contacts(force_message(now.nanoseconds, (BOOK, LEFT, 3.)))

    assert history(node) == (manipulation_node.ForceSample(now.nanoseconds, 7.),)
    assert node._adaptive_overload_latched is None


def test_duplicate_pair_inside_one_message_is_not_counted_twice():
    node, now = contact_node()
    node._on_contacts(force_message(now.nanoseconds, (LEFT, BOOK, 5.), (BOOK, LEFT, 5.)))

    assert history(node) == (manipulation_node.ForceSample(now.nanoseconds, 5.),)
    assert node._adaptive_overload_latched is None


def test_reordered_older_message_preserves_newer_history_and_contact_freshness():
    node, now = contact_node()
    stamps = [now.nanoseconds - offset for offset in (90_000_000, 60_000_000, 30_000_000)]
    for index in (0, 2, 1):
        node._on_contacts(force_message(stamps[index], (LEFT, BOOK, index + 1.)))

    assert history(node) == tuple(
        manipulation_node.ForceSample(stamp, index + 1.) for index, stamp in enumerate(stamps))
    assert node._left_target_contact_ns == stamps[-1]


@pytest.mark.parametrize('pair_forces', [(8.01,), (4.1, 4.1)])
def test_genuine_overload_still_latches_and_survives_contact_window_clear(pair_forces):
    node, now = contact_node()
    for name, force in zip((LEFT, LEFT_INNER), pair_forces):
        node._on_contacts(force_message(now.nanoseconds, (name, BOOK, force)))
    assert history(node)[-1].force_newtons == pytest.approx(sum(pair_forces))
    assert node._adaptive_overload_latched == 'force_overload'

    node._clear_target_contact_samples()
    assert node._adaptive_overload_latched == 'force_overload'


def test_new_contact_epoch_does_not_reuse_old_pair_maximum():
    node, now = contact_node()
    node._on_contacts(force_message(now.nanoseconds, (LEFT, BOOK, 6.)))
    node._clear_target_contact_samples()
    node._target_book_model = MODEL
    node._on_contacts(force_message(now.nanoseconds, (LEFT, BOOK, 2.)))

    assert history(node) == (manipulation_node.ForceSample(now.nanoseconds, 2.),)


def test_history_retains_only_latest_64_frames_despite_late_old_delivery():
    node, now = contact_node()
    stamps = [now.nanoseconds - (70 - index) * 1_000_000 for index in range(70)]
    for stamp in stamps:
        node._on_contacts(force_message(stamp, (LEFT, BOOK, 1.)))
    node._on_contacts(force_message(stamps[0], (LEFT, BOOK, 2.)))

    assert history(node) == tuple(
        manipulation_node.ForceSample(stamp, 1.) for stamp in stamps[-64:])


def pressure_node(latest_offset_ns):
    node, now = contact_node()
    node.grasp_max_position = .05
    node.adaptive_contact_force_minimum = .05
    node.adaptive_contact_samples = 3
    node.adaptive_contact_max_age = .15
    node.adaptive_contact_max_gap = .08
    node.adaptive_contact_min_span = .05
    node.adaptive_contact_max_skew = .04
    node.adaptive_velocity_tolerance = .01
    node.adaptive_effort_maximum = 8.
    node.adaptive_effort_delta_maximum = 4.
    node._latest_gripper_feedback = lambda: manipulation_node.GripperFeedback(
        now.nanoseconds, .018, 0., .2)
    latest = now.nanoseconds + latest_offset_ns
    stamps = [latest - offset for offset in (60_000_000, 30_000_000, 0)]
    for stamp in stamps:
        node._on_contacts(force_message(stamp, (LEFT, BOOK, 3.), (RIGHT, BOOK, 3.)))
    return node, now, stamps


def install_clock_progress(monkeypatch, on_sleep):
    wall = SimpleNamespace(value=0., sleeps=[])

    def sleep(duration):
        wall.value += duration
        wall.sleeps.append(duration)
        on_sleep()

    monkeypatch.setattr(manipulation_node, 'time', SimpleNamespace(
        monotonic=lambda: wall.value, sleep=sleep))
    return wall


@pytest.mark.parametrize('lag_ns', [40_000_000, 100_000_000])
def test_pressure_waits_for_clock_without_rewriting_or_chasing_snapshot(
    monkeypatch, lag_ns,
):
    node, now, stamps = pressure_node(lag_ns)
    inserted = []

    def clock_tick():
        now.nanoseconds += 20_000_000
        if not inserted:
            # Simulate a new producer sample arriving while the earlier fixed
            # snapshot waits for /clock; that newer sample must not be chased.
            future = stamps[-1] + 30_000_000
            node._on_contacts(force_message(future, (LEFT, BOOK, 3.), (RIGHT, BOOK, 3.)))
            inserted.append(future)

    wall = install_clock_progress(monkeypatch, clock_tick)
    evidence, model, identity = node._adaptive_pressure_evidence(
        minimum_width=.016, baseline_effort=.2)

    assert evidence.verified
    assert evidence.left_samples == evidence.right_samples == 3
    assert evidence.left_force == evidence.right_force == 3.
    assert model == MODEL and identity is None
    assert now.nanoseconds == stamps[-1]
    assert wall.sleeps and wall.value <= .5
    assert [sample.stamp_ns for sample in history(node)] == stamps + inserted


@pytest.mark.parametrize('lag_ns,wait_expected', [(40_000_000, True), (100_000_001, False)])
def test_stopped_or_far_future_clock_rejects_pressure_without_force_retimestamping(
    monkeypatch, lag_ns, wait_expected,
):
    node, now, stamps = pressure_node(lag_ns)
    original_now = now.nanoseconds
    wall = install_clock_progress(monkeypatch, lambda: None)
    evidence, _, _ = node._adaptive_pressure_evidence(minimum_width=.016, baseline_effort=.2)

    assert not evidence.verified
    assert evidence.reason == 'force_history_invalid'
    assert bool(wall.sleeps) is wait_expected
    assert wall.value <= .502
    assert now.nanoseconds == original_now
    assert [sample.stamp_ns for sample in history(node)] == stamps


def test_overload_arriving_during_clock_wait_cannot_be_hidden_by_older_snapshot(monkeypatch):
    node, now, stamps = pressure_node(40_000_000)

    def overload_event():
        node._on_contacts(force_message(stamps[-1], (LEFT, BOOK, 8.01)))
        now.nanoseconds = stamps[-1]

    wall = install_clock_progress(monkeypatch, overload_event)
    evidence, _, _ = node._adaptive_pressure_evidence(minimum_width=.016, baseline_effort=.2)

    assert not evidence.verified
    assert evidence.reason == 'force_overload'
    assert node._adaptive_overload_latched == 'force_overload'
    assert len(wall.sleeps) == 1


def test_pressure_reads_clock_after_joint_feedback_snapshot(monkeypatch):
    node, now, _ = pressure_node(0)
    feedback_reads = []

    def concurrent_feedback():
        # A JointState and /clock callback complete while the snapshot is being
        # copied. This sample is current, not future data or a clock failure.
        now.nanoseconds += 2_000_000
        feedback_reads.append(now.nanoseconds)
        return manipulation_node.GripperFeedback(now.nanoseconds, .018, 0., .2)

    node._latest_gripper_feedback = concurrent_feedback
    wall = install_clock_progress(monkeypatch, lambda: None)
    evidence, _, _ = node._adaptive_pressure_evidence(
        minimum_width=.016, baseline_effort=.2)

    assert evidence.verified
    assert len(feedback_reads) == 1
    assert not wall.sleeps


def test_pressure_does_not_replace_feedback_after_waiting_for_force_clock(monkeypatch):
    node, now, stamps = pressure_node(40_000_000)
    initial_feedback = manipulation_node.GripperFeedback(now.nanoseconds, .018, 0., .2)
    feedback = SimpleNamespace(latest=initial_feedback, reads=[])

    def latest_feedback():
        feedback.reads.append(feedback.latest)
        return feedback.latest

    node._latest_gripper_feedback = latest_feedback

    def concurrent_feedback():
        now.nanoseconds = stamps[-1]
        feedback.latest = manipulation_node.GripperFeedback(
            now.nanoseconds + 2_000_000, .018, 0., .2)

    wall = install_clock_progress(monkeypatch, concurrent_feedback)
    evidence, _, _ = node._adaptive_pressure_evidence(
        minimum_width=.016, baseline_effort=.2)

    assert evidence.verified
    assert feedback.reads == [initial_feedback]
    assert len(wall.sleeps) == 1
    assert feedback.latest.stamp_ns > now.nanoseconds


def test_pressure_wait_target_includes_fixed_joint_feedback_stamp(monkeypatch):
    node, now, _ = pressure_node(0)
    future_stamp = now.nanoseconds + 20_000_000
    node._latest_gripper_feedback = lambda: manipulation_node.GripperFeedback(
        future_stamp, .018, 0., .2)

    def clock_tick():
        now.nanoseconds = future_stamp

    wall = install_clock_progress(monkeypatch, clock_tick)
    evidence, _, _ = node._adaptive_pressure_evidence(
        minimum_width=.016, baseline_effort=.2)

    assert evidence.verified
    assert len(wall.sleeps) == 1
    assert now.nanoseconds == future_stamp


def test_unstamped_contacts_use_clock_fallback_without_changing_joint_stamp_default():
    node, now = contact_node()
    message = force_message(0, (LEFT, BOOK, 2.))
    node._on_contacts(message)
    first = now.nanoseconds
    now.nanoseconds += 30_000_000
    node._on_contacts(message)

    assert history(node) == (
        manipulation_node.ForceSample(first, 2.),
        manipulation_node.ForceSample(now.nanoseconds, 2.),
    )
    future = force_message(now.nanoseconds + 40_000_000)
    # The contact opt-out must not alter the previous helper default used by
    # joint feedback; only raw producer contact stamps need the separate path.
    assert manipulation_node._message_stamp_ns(future, now.nanoseconds) == now.nanoseconds
