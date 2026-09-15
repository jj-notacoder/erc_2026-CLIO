"""Explicit release-pose completion evidence; never a hand-return milestone.

The caller selects this class from its local immutable attempt policy. A status
payload cannot switch an ordinary ReleaseEvidence attempt to this policy.
The inherited clock, deadline and exact post-open contact window are unchanged.
"""
from .release_evidence import ReleaseEvidence, stamp_valid
from .place_contact_guard import collision_pair, producer_stamp
# Opening handoffs retain semantic witnesses, not message-count queues.


MODE = 'release_pose'
EVENT = 'placement_release_pose_measured'
CONTRACT = 'stationary_release_pose_received_events_v2'
BASIS = 'raw_stationary_release_pose_and_positive_named_bin_contact_window'


def contract_fields():
    """Received-event veto is not stream coverage, detachment or containment."""
    return dict(release_evidence_contract=CONTRACT, basis=BASIS,
        contact_publication_policy='nonempty_events_only',
        received_contact_fault_latch_clear=True,
        contact_stream_coverage_verified=False, detachment_verified=False,
        hand_clear_verified=False, physical_inside_verified=None)


def contract_matches(payload):
    expected = contract_fields()
    return (all(key in payload and type(payload[key]) is type(value)
                and payload[key] == value for key,value in expected.items())
            and not any(key in payload for key in ('raw_contact_start_ns',
                'raw_contact_last_ns', 'no_observed_target_robot_contact_verified')))


def received_contact_event(identity, epoch, previous, stamp, now, pairs, topic):
    """Validate a received event only. No callback is promised in free space."""
    if topic not in ('/contacts', '/bin_contacts', '/table_contacts'):
        return 'release_pose_contact_source_invalid', False
    if not stamp_valid(stamp):
        return 'release_pose_contact_stamp_invalid', False
    if epoch is not None and stamp <= epoch:
        return None, False
    if not stamp_valid(now) or not -100_000_000 <= now-stamp <= 250_000_000:
        return 'release_pose_contact_not_fresh', False
    if previous is not None and stamp < previous:
        return 'release_pose_contact_stamp_reversed', False
    if not isinstance(pairs, (tuple, list)):
        return 'release_pose_contact_pair_invalid', False
    exact_bin = False
    for pair in pairs:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or not all(isinstance(name, str) and name for name in pair)):
            return 'release_pose_contact_pair_invalid', False
        first, second = pair
        if collision_pair(first, second) is not None:
            return 'release_pose_scene_contact', False
        for target, other in ((first, second), (second, first)):
            if identity.target_model not in target.split('::')[:-2]:
                continue
            if 'tiago_pro' in other.split('::')[:-2]:
                return 'release_pose_target_robot_contact', False
            exact_bin |= 'erc_collection_bin' in other.split('::')[:-2]
    return None, bool(exact_bin and topic == '/bin_contacts')


def positive_window(stamps, epoch, now):
    """Serialize the same current suffix used by ReleaseEvidence.evaluate."""
    run = []
    for stamp in sorted(s for s in stamps if epoch < s <= now and s >= now-750_000_000):
        if run and stamp-run[-1] > 125_000_000:
            run = []
        run.append(stamp)
    return run


def measurement_at_admission(payload, epoch):
    """Actual producer admission time is distinct from delayed DDS receipt."""
    now = payload.get('release_measurement_stamp_ns')
    joint = payload.get('producer_stamp_ns')
    odom = payload.get('odom_producer_stamp_ns')
    start = payload.get('bin_contact_start_ns')
    last = payload.get('bin_contact_last_ns')
    count = payload.get('bin_contact_samples')
    gap = payload.get('bin_contact_max_gap_ns')
    return (all(stamp_valid(v) for v in (now,joint,odom,start,last))
        and (epoch is None or epoch < start)
        and all(0 <= now-v <= 150_000_000 for v in (joint,odom,last))
        and 500_000_000 <= last-start and now-start <= 750_000_000
        and type(count) is int and 3 <= count <= 1024
        and type(gap) is int and 0 < gap <= 125_000_000
        and last-start <= (count-1)*gap)


def checked_enabled(value):
    if type(value) is not bool:
        raise ValueError('place_finish_at_release_enabled must be boolean')
    return value


class PendingContactEvents:
    """Bounded veto witnesses while the verified opening epoch is unknown.

    For each fixed source/reason, max(stamp) > epoch iff at least one received
    witness is post-opening. Benign messages cannot overwrite a veto. Unknown
    stamps remain a separate latch because no later epoch can classify them.
    Only exact positive bin stamps need a history; duplicate events add none.
    This is neither a raw stream census nor evidence of absent collisions.
    """
    def __init__(self, identity):
        self.identity = identity
        self.last_stamps = {}
        self.witnesses = {}
        self.unknown = None
        self.bin_stamps = set()
        self.sequence = 0
        self.last_now = None
        self.resolved = False
        self.fault_witness = None

    def __len__(self):
        return len(self.witnesses) + len(self.bin_stamps) + (self.unknown is not None)

    def observe(self, stamp, now, pairs, topic, error=None):
        if self.resolved:
            raise ValueError('Opening contact history already resolved')
        self.sequence += 1
        # Keep a concrete relevant pair when the rejection has one. Malformed
        # data remains a reason, never a made-up collision pair.
        witness = dict(stamp_ns=stamp if stamp_valid(stamp) else None,
            receipt_ns=now if stamp_valid(now) else None, source_topic=topic,
            sequence=self.sequence, pair=None)
        if not stamp_valid(now) or (self.last_now is not None and now < self.last_now):
            error = 'invalid_or_reversed_clock'
        else:
            self.last_now = now
        if topic not in ('/contacts', '/bin_contacts', '/table_contacts'):
            error = 'release_pose_contact_source_invalid'
        if error or not stamp_valid(stamp):
            witness['reason'] = error or 'release_pose_contact_stamp_invalid'
            self.unknown = self.unknown or witness
            return
        reason, positive = received_contact_event(self.identity, None,
            self.last_stamps.get(topic), stamp, now, pairs, topic)
        if reason:
            witness['reason'] = reason
            if isinstance(pairs, (tuple, list)):
                for pair in pairs:
                    if (isinstance(pair, (tuple, list)) and len(pair) == 2
                            and all(isinstance(name, str) and name for name in pair)
                            and received_contact_event(self.identity, None, None,
                                stamp, now, (pair,), topic)[0] == reason):
                        witness['pair'] = tuple(pair)
                        break
            # Only the three known topics get separate buckets. Invalid source
            # strings cannot create unbounded dictionary keys.
            key = (topic if topic in ('/contacts','/bin_contacts','/table_contacts')
                   else '<invalid>', reason)
            old = self.witnesses.get(key)
            if old is None or stamp > old['stamp_ns']:
                self.witnesses[key] = witness
            return
        self.last_stamps[topic] = stamp
        self.bin_stamps = {s for s in self.bin_stamps if s >= now-750_000_000}
        if positive:
            self.bin_stamps.add(stamp)
            if len(self.bin_stamps) > 1024:
                witness['reason'] = 'contact_history_rate_overflow'
                self.unknown = self.unknown or witness
                self.bin_stamps.clear()

    def resolve(self, release, invalidate, last_received, now):
        """Transfer only actual post-epoch evidence under the caller's lock."""
        epoch = release.open_epoch
        if not stamp_valid(epoch):
            raise ValueError('Verified opening epoch required')
        if self.resolved:
            return
        self.resolved = True
        relevant = [w for w in self.witnesses.values() if w['stamp_ns'] > epoch]
        if self.unknown is not None:
            relevant.append(self.unknown)
        if relevant:
            # First retained concrete witness wins; later callbacks cannot
            # replace the owner's existing first-fault latch.
            self.fault_witness = min(relevant, key=lambda w:w['sequence'])
            invalidate(self.fault_witness['reason'])
        for topic, stamp in self.last_stamps.items():
            if stamp > epoch:
                last_received[topic] = stamp
        for stamp in sorted(self.bin_stamps):
            if stamp > epoch:
                release.contact(stamp, now, exact_target_bin_pair=True)
        if release.fault:
            invalidate(release.fault)
        self.witnesses.clear()
        self.bin_stamps.clear()
        self.unknown = None
        self.last_stamps.clear()


class ReleasePoseEvidence(ReleaseEvidence):
    """One attempt's measured stationary release pose and named bin contact."""

    def __init__(self, identity):
        super().__init__(identity)
        self.finish_measurement = None
        self.pending_finish_measurement = None
        self.terminal_stamp = None
        self.last_received_stamps = {}
        self.pending_contact_events = PendingContactEvents(identity)

    def opening(self, payload, now_ns):
        accepted = super().opening(payload, now_ns)
        if accepted:
            self.pending_contact_events.resolve(self, self.invalidate,
                self.last_received_stamps, now_ns)
        return accepted

    def _received(self, stamp, now_ns, pairs, topic, error=None):
        if self.fault:
            return
        if stamp_valid(stamp) and self.open_epoch is not None and stamp <= self.open_epoch:
            return
        if error:
            self.invalidate(error)
            return
        reason, positive = received_contact_event(self.identity, self.open_epoch,
            self.last_received_stamps.get(topic), stamp, now_ns, pairs, topic)
        if reason:
            self.invalidate(reason)
            return
        self.last_received_stamps[topic] = stamp
        if topic == '/bin_contacts':
            self.contact(stamp, now_ns, exact_target_bin_pair=positive)

    def observe_contact_message(self, message, now_ns, *, source_topic):
        if self.fault:
            return
        if not self._clock(now_ns):
            return
        stamp, pairs, error = None, (), None
        try:
            contacts = message.contacts
            if not isinstance(contacts, (tuple, list)):
                raise ValueError('Contact sequence missing')
            pairs = tuple((c.collision1.name, c.collision2.name) for c in contacts)
            stamp = producer_stamp(message)
        except (AttributeError, TypeError, ValueError, OverflowError):
            error = 'release_pose_contacts_malformed'
        if self.open_epoch is None:
            # Veto summaries preserve delayed post-open hazards without
            # retaining every unrelated sensor publication during PLACE.
            self.pending_contact_events.observe(stamp, now_ns, pairs, source_topic, error)
            return
        self._received(stamp, now_ns, pairs, source_topic, error)

    @property
    def completion_mode(self):
        return MODE

    def hand_return(self, payload, now_ns):
        if not self.identity.matches(payload):
            return False
        self.invalidate('unexpected_hand_return_in_release_pose_mode')
        return False

    def motion_terminal(self, payload, now_ns, wall_seconds):
        if not self.identity.matches(payload):
            return False
        terminal = payload.get('release_terminal_stamp_ns')
        if (payload.get('completion_mode') != MODE or payload.get('hand_return') is not False
                or not contract_matches(payload) or not stamp_valid(terminal)
                or not -100_000_000 <= now_ns-terminal <= 250_000_000
                or (self.terminal_stamp is not None and self.terminal_stamp != terminal)):
            self.invalidate('release_pose_terminal_mode_mismatch')
            return False
        self.terminal_stamp = terminal
        return super().motion_terminal(payload, now_ns, wall_seconds)

    def finish_pose(self, payload, now_ns):
        if not self.identity.matches(payload):
            return False
        if not self._clock(now_ns):
            return False
        stamp = payload.get('producer_stamp_ns')
        start = payload.get('stationary_start_ns')
        valid = (
            payload.get('event') == EVENT
            and payload.get('command') == 'place'
            and payload.get('completion_mode') == MODE
            and payload.get('verified') is True
            and payload.get('hand_return') is False
            and contract_matches(payload)
            and measurement_at_admission(payload, self.open_epoch)
            and stamp_valid(stamp) and stamp_valid(start)
            and stamp - start >= 100_000_000
        )
        admission = payload.get('release_measurement_stamp_ns')
        if (valid and -250_000_000 <= admission - now_ns <= 100_000_000
                and (admission > now_ns or self.open_epoch is None)):
            self.pending_finish_measurement = dict(payload)
            return False
        if (not valid or self.open_epoch is None
                or not self.open_epoch < start <= stamp <= admission <= now_ns
                or now_ns - admission > 250_000_000):
            self.invalidate('release_pose_unverified')
            return False
        if self.finish_measurement is not None and self.finish_measurement != payload:
            self.invalidate('release_pose_measurement_changed')
            return False
        self.finish_measurement = dict(payload)
        return True

    def evaluate(self, now_ns, wall_seconds):
        # The inherited implementation owns all clock/deadline/contact checks.
        result = super().evaluate(now_ns, wall_seconds)
        pending = self.pending_finish_measurement
        if (pending is not None and self.open_epoch is not None
                and pending['release_measurement_stamp_ns'] <= now_ns):
            self.pending_finish_measurement = None
            self.finish_pose(pending, now_ns)
        result.update(completion_mode=MODE, hand_return=False,
                      hand_return_measured=False,
                      release_pose_measured=self.finish_measurement is not None)
        result.update(contract_fields())
        result['received_contact_fault_latch_clear'] = self.fault is None
        if self.fault:
            result.update(status='invalid', reason=self.fault,
                          operation_completed=False)
            return result
        measured = self.finish_measurement
        terminal_current = bool(measured is not None and self.terminal_stamp is not None
            and measured['release_measurement_stamp_ns'] == self.terminal_stamp <= now_ns
            and all(0 <= self.terminal_stamp-measured[k] <= 150_000_000 for k in
                ('producer_stamp_ns','odom_producer_stamp_ns','bin_contact_last_ns'))
            and self.terminal_stamp-measured['bin_contact_start_ns'] <= 750_000_000)
        if (measured is not None and self.terminal_stamp is not None
                and self.terminal_stamp <= now_ns and not terminal_current):
            self.invalidate('release_pose_terminal_measurement_stale')
            result.update(status='invalid', reason=self.fault, operation_completed=False,
                          received_contact_fault_latch_clear=False)
            return result
        if (result['status'] == 'pending'
                and self.verification_start is not None
                and result['post_release_contact_verified']
                and terminal_current):
            result.update(status='operation_completed', reason=None,
                          operation_completed=True,
                          success_scope='placement_operation_completed',
                          delivery_outcome='released_with_bin_contact')
        return result
