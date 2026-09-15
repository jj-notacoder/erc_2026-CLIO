"""Raw sensor conditions for release-pose hold; no runtime motion fixture."""
import pytest

from erc_phase1_solution.release_evidence import AttemptIdentity
from erc_phase1_solution.release_pose_hold_evidence import ReleasePoseHoldEvidence


TARGET = 'book_col_3_row_2_red'
BOOK = TARGET+'::book_link::book_collision'
BIN = 'erc_collection_bin::bin_link::bin_collision'
ROBOT = 'tiago_pro::gripper_left_fingertip_left_link::collision'
NAMES = ('torso_lift_joint', 'arm_left_1_joint')


def sample(stamp):
    return dict(producer_stamp_ns=stamp,
        positions=dict(torso_lift_joint=.3, arm_left_1_joint=.4,
                       gripper_left_finger_joint=.069),
        velocities={n: 0. for n in (*NAMES, 'gripper_left_finger_joint')},
        odom=dict(stamp_ns=stamp, linear_speed=0., angular_speed=0.))


def fixture():
    identity = AttemptIdentity('trial', 'attempt', TARGET)
    evidence = ReleasePoseHoldEvidence(identity, (.3,.4), NAMES, .069)
    evidence.opening(dict(vars(identity),event='placement_open_measured',
        verified=True,producer_stamp_ns=1_000_000_000),1_000_000_000)
    return evidence


def feed(evidence, *, robot_contact=False, global_frames=True):
    for stamp in range(1_050_000_000,1_600_000_001,50_000_000):
        pairs=[(BOOK,BIN)]
        if robot_contact and stamp==1_300_000_000:
            pairs.append((ROBOT,BOOK))
        evidence.observe_contacts(stamp,stamp,pairs,source_topic='/bin_contacts')
        if global_frames:
            evidence.observe_contacts(stamp,stamp,pairs,source_topic='/contacts')
        evidence.observe_joint_sample(sample(stamp),stamp,actions_active=False)


def test_current_open_stationary_pose_and_both_raw_windows_are_required():
    evidence=fixture();feed(evidence)
    assert evidence.evaluate(1_600_000_000,1.)['ready']
    payload=evidence.measurement(1_600_000_000,1.)
    assert payload['event']=='placement_release_pose_measured'
    assert payload['hand_return'] is payload['hand_clear_verified'] is False
    assert payload['physical_inside_verified'] is None


def test_target_robot_pair_after_open_latches_even_when_contact_disappears():
    evidence=fixture();feed(evidence,robot_contact=True)
    assert evidence.fault=='release_pose_target_robot_contact'
    assert not evidence.evaluate(1_600_000_000,1.)['ready']


def test_bin_contacts_alone_do_not_prove_no_raw_target_robot_contacts():
    evidence=fixture();feed(evidence,global_frames=False)
    result=evidence.evaluate(1_600_000_000,1.)
    assert result['post_release_contact_verified']
    # Positive named bin events plus continuous stopped feedback are sufficient
    # for this operation contract; they never attest negative stream coverage.
    assert result['received_contact_fault_latch_clear']
    assert result['contact_stream_coverage_verified'] is False
    assert result['ready']


@pytest.mark.parametrize('fault', ['missing_velocity','moving_arm','not_open','active_goal'])
def test_between_poll_feedback_fault_is_latched(fault):
    evidence=fixture();feed(evidence)
    raw=sample(1_650_000_000)
    if fault=='missing_velocity':del raw['velocities']['arm_left_1_joint']
    if fault=='moving_arm':raw['velocities']['arm_left_1_joint']=.002
    if fault=='not_open':raw['positions']['gripper_left_finger_joint']=.06
    evidence.observe_joint_sample(raw,1_650_000_000,actions_active=fault=='active_goal')
    evidence.observe_joint_sample(sample(1_700_000_000),1_700_000_000,actions_active=False)
    assert evidence.fault is not None
    assert not evidence.evaluate(1_700_000_000,1.)['ready']


def test_preopen_contact_cannot_poison_or_satisfy_postopen_window():
    evidence=fixture()
    evidence.observe_contacts(1_000_000_000,1_000_000_000,[(BOOK,ROBOT)],source_topic='/contacts')
    assert evidence.fault is None
    assert not evidence.evaluate(1_000_000_000,1.)['ready']


def test_stale_joint_feedback_and_duplicate_frames_cannot_complete():
    evidence=fixture();feed(evidence)
    assert not evidence.evaluate(1_800_000_000,1.)['ready']
    evidence=fixture()
    for _ in range(100):
        evidence.observe_contacts(1_100_000_000,1_100_000_000,[(BOOK,BIN)],source_topic='/contacts')
    assert len(evidence.raw_contact_frames)==1
    assert not evidence.evaluate(1_100_000_000,1.)['ready']


def test_different_named_book_is_not_the_attempt_contact():
    evidence=fixture()
    for stamp in range(1_050_000_000,1_600_000_001,50_000_000):
        evidence.observe_contacts(stamp,stamp,[(BOOK.replace('_red','_blue'),BIN)],source_topic='/contacts')
        evidence.observe_joint_sample(sample(stamp),stamp,actions_active=False)
    result=evidence.evaluate(1_600_000_000,1.)
    assert result['received_contact_fault_latch_clear']
    assert result['contact_stream_coverage_verified'] is False
    assert not result['post_release_contact_verified']
    assert not result['ready']
