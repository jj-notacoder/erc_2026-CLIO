"""Bounded raw observations for an owned, motionless post-open hold.

This module issues no commands. Contact messages are nonempty events, not a
heartbeat or complete census. Positive exact bin contacts form the contact
window; received hazards remain latched and silence proves nothing.
The adapter must independently preserve the checked scene and command owner.
"""
from collections import deque
import math

from .release_evidence import ReleaseEvidence, measured_pose_is_open, stamp_valid
from .release_pose_evidence import (EVENT, MODE, contract_fields,
    received_contact_event, positive_window)


def _model(name, model):
    return isinstance(name, str) and model in name.split('::')[:-2]


class ReleasePoseHoldEvidence:
    def __init__(self, identity, goal, joint_names, open_position):
        self.identity = identity
        self.goal = tuple(goal)
        self.joint_names = tuple(joint_names)
        if (len(self.goal) != len(self.joint_names)
                or not all(math.isfinite(v) for v in self.goal)
                or not math.isfinite(open_position)):
            raise ValueError('Finite checked release goal required')
        self.open_position = open_position
        self.release = ReleaseEvidence(identity)
        self.raw_contact_frames = deque(maxlen=1024)
        self.last_raw_contact_stamp = None
        self.raw_contact_history_truncated = False
        self.last_received_stamps = {}
        self.last_joint_stamp = None
        self.stationary_start = None
        self.latest_sample = None
        self.fault = None

    def invalidate(self, reason):
        self.fault = self.fault or str(reason)

    def opening(self, payload, now_ns):
        accepted = self.release.opening(payload, now_ns)
        if not accepted:
            self.invalidate('release_pose_opening_unverified')
        return accepted

    def observe_contacts(self, stamp_ns, now_ns, pairs, *, source_topic):
        if self.fault or self.release.open_epoch is None:
            return
        reason, positive = received_contact_event(self.identity, self.release.open_epoch,
            self.last_received_stamps.get(source_topic), stamp_ns, now_ns, pairs, source_topic)
        if reason:
            self.invalidate(reason)
            return
        if stamp_ns <= self.release.open_epoch:
            return
        self.last_received_stamps[source_topic] = stamp_ns
        if source_topic == '/bin_contacts':
            self.release.contact(stamp_ns, now_ns, exact_target_bin_pair=positive)
        if source_topic != '/contacts':
            return
        # Bounded received-event diagnostics only, never admission coverage.
        self.last_raw_contact_stamp = stamp_ns
        while self.raw_contact_frames and self.raw_contact_frames[0] < now_ns-750_000_000:
            self.raw_contact_frames.popleft()
        if not self.raw_contact_frames or self.raw_contact_frames[-1] != stamp_ns:
            # Every event has already passed the unchanged veto checks. This
            # ring is diagnostics only; bounded eviction is not lost coverage.
            if len(self.raw_contact_frames) == self.raw_contact_frames.maxlen:
                self.raw_contact_history_truncated = True
            self.raw_contact_frames.append(stamp_ns)

    def observe_joint_sample(self, raw, now_ns, *, actions_active):
        if self.fault or self.release.open_epoch is None:
            return
        stamp = raw.get('producer_stamp_ns')
        if stamp_valid(stamp) and stamp <= self.release.open_epoch:
            return
        valid, reason = measured_pose_is_open(
            raw, raw.get('odom'), self.goal, now_ns, self.joint_names,
            pending_negative_check=True, open_position=self.open_position)
        if actions_active or not valid:
            self.invalidate('release_pose_action_active' if actions_active else reason)
            return
        if (self.last_joint_stamp is not None
                and not 0 <= stamp-self.last_joint_stamp <= 75_000_000):
            self.invalidate('release_pose_joint_gap_or_reversal')
            return
        self.last_joint_stamp = stamp
        if self.stationary_start is None:
            self.stationary_start = stamp
        self.latest_sample = dict(raw, positions=dict(raw['positions']),
                                  velocities=dict(raw['velocities']),
                                  odom=dict(raw['odom']))

    def evaluate(self, now_ns, wall_seconds):
        result = self.release.evaluate(now_ns, wall_seconds)
        if result['status'] in ('invalid', 'expired'):
            self.invalidate(result['reason'])
        result.update(ready=False, reason=self.fault, **contract_fields())
        result['received_contact_fault_latch_clear'] = self.fault is None
        if self.fault or self.release.open_epoch is None:
            return result
        sample = self.latest_sample
        stationary = bool(sample is not None and self.stationary_start is not None
            and self.last_joint_stamp-self.stationary_start >= 100_000_000
            and measured_pose_is_open(sample, sample['odom'], self.goal, now_ns,
                self.joint_names, open_position=self.open_position)[0])
        result['ready'] = bool(stationary and result['post_release_contact_verified'])
        return result

    def measurement(self, now_ns, wall_seconds):
        result = self.evaluate(now_ns, wall_seconds)
        if not result['ready']:
            raise RuntimeError(result['reason'] or 'release_pose_hold_not_verified')
        raw = self.latest_sample
        window = positive_window(self.release.contact_stamps, self.release.open_epoch, now_ns)
        assert len(window) == result['contact_samples']
        return dict(vars(self.identity), event=EVENT, command='place', verified=True,
            completion_mode=MODE, hand_return=False,
            release_measurement_stamp_ns=now_ns,
            producer_stamp_ns=self.last_joint_stamp,
            odom_producer_stamp_ns=raw['odom']['stamp_ns'],
            stationary_start_ns=self.stationary_start,
            master_position=raw['positions']['gripper_left_finger_joint'],
            master_velocity=raw['velocities']['gripper_left_finger_joint'],
            measured_joint_positions=[raw['positions'][n] for n in self.joint_names],
            bin_contact_start_ns=window[0], bin_contact_last_ns=window[-1],
            bin_contact_samples=len(window),
            bin_contact_max_gap_ns=max(b-a for a,b in zip(window,window[1:])),
            **contract_fields())
