"""Owned normal PLACE completion at its checked release pose; no actuator calls."""
from contextlib import nullcontext
from collections import deque
from dataclasses import dataclass, field
import json
import math
import time
from types import SimpleNamespace

from .motion_profiles import IK_JOINTS
from .place_contact_guard import PlaceContactGuard, producer_stamp, collision_pair
from .placement_scene_context import measured_scene_context
from .release_evidence import AttemptIdentity, stamp_valid, measured_pose_is_open
from .release_pose_evidence import (EVENT, MODE, CONTRACT, contract_fields, checked_enabled,
    PendingContactEvents)
from .release_pose_hold_evidence import ReleasePoseHoldEvidence


class ReleasePoseFeedbackStale(RuntimeError):
    """Original rejection reason with a frozen scalar snapshot, never admission."""
    def __init__(self, details):
        super().__init__('release_pose_hold_feedback_stale')
        self._detail_items = tuple(details.items())

    @property
    def details(self):
        return dict(self._detail_items)


@dataclass(frozen=True)
class _Binding:
    identity: AttemptIdentity
    goal: tuple
    scene: object
    scene_bytes: str
    guard: PlaceContactGuard
    chains: tuple


@dataclass
class _Owner:
    binding: _Binding
    evidence: ReleasePoseHoldEvidence
    phase: str = 'opening'
    open_wall: float | None = None
    last_ros: int | None = None
    last_joint_sequence: int | None = None
    opening_contacts: PendingContactEvents = field(init=False)

    def __post_init__(self):
        self.opening_contacts = PendingContactEvents(self.binding.identity)


def validate_mode(node, payload):
    enabled = checked_enabled(getattr(node, 'place_finish_at_release_enabled', False))
    requested = (payload or {}).get('completion_mode')
    if (requested != (MODE if enabled else None)
            or (payload or {}).get('release_evidence_contract') != (CONTRACT if enabled else None)):
        raise RuntimeError('release_pose_completion_mode_mismatch')


def post_open(node):
    owner = getattr(node, '_release_pose_owner', None)
    return isinstance(owner, _Owner) and owner.evidence.release.open_epoch is not None


def block_postopen_gripper_hold(node):
    """Caller owns the command lock; post-open faults cannot reclose the hand.

    Failed-open cleanup may restore the old payload monitor. Keep its actual
    sender inert after measured release, even after the owner is deactivated.
    """
    with node._lock:
        if (not post_open(node)
                and getattr(node, '_release_pose_fault_latched', None) is None):
            return False
        node._cancel.set()
        owner = getattr(node, '_release_pose_owner', None)
        if isinstance(owner, _Owner) and owner.phase != 'terminal':
            owner.evidence.invalidate('release_pose_postopen_hold_rejected')
        return True


def prepare(node, correlation, goal, direct_empty_home, *, release_only_endpoint=None):
    if release_only_endpoint is not None:
        from .release_only_place_planning import Endpoint
        if type(release_only_endpoint) is not Endpoint or direct_empty_home is not None:
            raise RuntimeError('release-only finish requires its typed endpoint capability')
        release_only_endpoint.require(node, correlation, goal)
    if not checked_enabled(getattr(node, 'place_finish_at_release_enabled', False)):
        return None
    if (not getattr(node, 'delivery_evidence_enabled', False)
            or not getattr(node, 'table_scene_required', False)
            or not getattr(node, 'bin_scene_required', False)
            # The original checked direct-fold result is exactly []; it is
            # a success marker, not the checked return-goal list. None marks
            # return-through-carry. Other shapes are outside this contract.
            or (release_only_endpoint is None
                and (type(direct_empty_home) is not list or len(direct_empty_home) != 0))):
        raise RuntimeError('release_pose_requires_complete_checked_normal_return_plan')
    identity = AttemptIdentity(**correlation)
    reference = getattr(node, '_active_place_scene_reference', None)
    if reference is None:
        raise RuntimeError('release_pose_registered_scene_missing')
    measured_scene_context(node, reference)
    evidence = ReleasePoseHoldEvidence(identity, goal, IK_JOINTS, node.gripper_open)
    with node._adaptive_command_guard():
        with node._lock:
            if release_only_endpoint is not None:
                release_only_endpoint.request.require_locked()
            guard = getattr(node, '_place_contact_guard', None)
            if (not isinstance(guard, PlaceContactGuard)
                    or not identity.matches(guard.correlation) or guard.fault is not None
                    or type(guard.started_ns) is not int or guard.started_ns <= 0
                    or node._target_book_model != identity.target_model
                    or node._held_book_corners is None or not node._busy
                    or node._cancel.is_set() or node._goal_handles
                    or getattr(node, '_pending_retained_acceptances', ())
                    or getattr(node, '_release_pose_owner', None) is not None
                    or getattr(node, '_delivery_measurement_active', False)
                    or node._active_place_scene_reference is not reference):
                raise RuntimeError('release_pose_owner_admission_failed')
            binding = _Binding(identity, tuple(evidence.goal), reference,
                json.dumps(reference, sort_keys=True, allow_nan=False), guard,
                (node.chain, node.right_chain, node.head_chain))
            owner = _Owner(binding, evidence)
            node._delivery_raw_samples = deque(maxlen=128)
            node._delivery_sample_sequence = 0
            node._delivery_raw_odom = None
            node._release_pose_owner = owner
            return owner


def remember_open(node, owner, fields):
    if owner is None:
        return fields.get('verified') is True
    with node._lock:
        if getattr(node, '_release_pose_owner', None) is not owner:
            raise RuntimeError('release_pose_owner_changed_during_open')
        now = int(node.get_clock().now().nanoseconds)
        if not owner.evidence.opening(dict(fields, event='placement_open_measured'), now):
            node._cancel.set()
            return False
        owner.open_wall = time.monotonic()
        owner.last_ros = now
        owner.phase = 'opened'
        # The producer's verified opening sample is the exact handoff anchor.
        # A bounded ring may discard older PRE-opening data; it must retain
        # this sample and every subsequent callback through this locked read.
        sequence = fields.get('raw_sample_sequence')
        history = list(node._delivery_raw_samples)
        epoch = owner.evidence.release.open_epoch
        anchors = [i for i, raw in enumerate(history)
                   if raw.get('sequence') == sequence
                   and raw.get('producer_stamp_ns') == epoch]
        latest = node._delivery_sample_sequence
        tail = history[anchors[0]:] if len(anchors) == 1 else []
        if (type(sequence) is not int or sequence <= 0
                or type(latest) is not int or latest < sequence
                or not tail or len(tail) != latest-sequence+1
                or any(type(raw.get('sequence')) is not int
                       or raw['sequence'] != sequence+i for i, raw in enumerate(tail))
                or not measured_pose_is_open(tail[0], tail[0].get('odom'),
                    owner.evidence.goal, now, IK_JOINTS, pending_negative_check=True,
                    open_position=node.gripper_open)[0]):
            owner.evidence.invalidate('release_pose_joint_handoff_gap')
            node._cancel.set()
            return False
        owner.last_joint_sequence = sequence
        owner.evidence.last_joint_stamp = epoch
        # Status publication and callback scheduling may interleave. Replay
        # preserved raw observations newer than the measured opening epoch.
        owner.opening_contacts.resolve(owner.evidence.release,
            owner.evidence.invalidate, owner.evidence.last_received_stamps, now)
        for raw in tail[1:]:
            observe_joint_locked(node, raw)
        if owner.evidence.fault:
            node._cancel.set()
            return False
        return True


def observe_joint_locked(node, raw):
    """Called under the existing raw-adapter sensor lock, before normalization."""
    owner = getattr(node, '_release_pose_owner', None)
    if not isinstance(owner, _Owner) or owner.phase == 'terminal':
        return
    epoch = owner.evidence.release.open_epoch
    stamp = raw.get('producer_stamp_ns')
    if epoch is not None:
        sequence = raw.get('sequence')
        if (type(sequence) is not int or owner.last_joint_sequence is None
                or sequence != owner.last_joint_sequence+1):
            owner.evidence.invalidate('release_pose_joint_sequence_gap')
        else:
            owner.last_joint_sequence = sequence
    if epoch is not None and stamp_valid(stamp) and stamp > epoch:
        for name, goal in owner.binding.scene['parked_joints'].items():
            q = raw.get('positions', {}).get(name, math.nan)
            v = raw.get('velocities', {}).get(name, math.nan)
            if (not math.isfinite(q) or not math.isfinite(v)
                    or abs(q-goal) > .001 or abs(v) > .001):
                owner.evidence.invalidate('release_pose_parked_feedback_changed')
    owner.evidence.observe_joint_sample(raw, int(node.get_clock().now().nanoseconds),
        actions_active=bool(node._goal_handles or getattr(node, '_pending_retained_acceptances', ())))
    if owner.evidence.fault:
        node._cancel.set()


def _contacts_locked(owner, stamp, now, pairs, topic):
    epoch = owner.evidence.release.open_epoch
    if (epoch is not None and (stamp is None or stamp > epoch)
            and any(collision_pair(a, b) is not None for a,b in pairs)):
        # Same sensor lock as terminal admission; the existing guard also
        # records the original detailed diagnostic outside this function.
        owner.evidence.invalidate('release_pose_scene_contact')
    owner.evidence.observe_contacts(stamp, now, pairs, source_topic=topic)


def observe_contacts(node, message, source_topic):
    with node._lock:
        owner = getattr(node, '_release_pose_owner', None)
        if not isinstance(owner, _Owner) or owner.phase == 'terminal':
            return
        if source_topic not in ('/contacts', '/bin_contacts', '/table_contacts'):
            owner.evidence.invalidate('release_pose_contact_source_invalid')
        else:
            try:
                contacts = message.contacts
                if not isinstance(contacts, (list, tuple)):
                    raise ValueError('Contact sequence missing or malformed')
                pairs = [(getattr(getattr(c, 'collision1', None), 'name', None),
                          getattr(getattr(c, 'collision2', None), 'name', None)) for c in contacts]
                stamp = producer_stamp(message)
                now = int(node.get_clock().now().nanoseconds)
                if owner.evidence.release.open_epoch is None:
                    owner.opening_contacts.observe(stamp, now, tuple(pairs), source_topic)
                else:
                    _contacts_locked(owner, stamp, now, pairs, source_topic)
            except (AttributeError, TypeError, ValueError, OverflowError):
                owner.evidence.invalidate('release_pose_contacts_malformed')
        if owner.evidence.fault:
            node._cancel.set()


def _scene_snapshot_locked(node):
    return (dict(node.joints), dict(getattr(node, '_joint_stamps_ns', {})),
            dict(getattr(node, '_staging_odom', None) or {}))


def _require_owner_locked(node, owner, now, wall):
    binding = owner.binding
    if (getattr(node, '_release_pose_owner', None) is not owner
            or owner.phase not in ('opened', 'holding', 'ready')
            or node._active_place_scene_reference is not binding.scene
            or json.dumps(binding.scene, sort_keys=True, allow_nan=False) != binding.scene_bytes
            or getattr(node, '_place_contact_guard', None) is not binding.guard
            or any(current is not prior for current, prior in
                   zip((node.chain, node.right_chain, node.head_chain), binding.chains))
            or binding.guard.fault is not None
            or not binding.identity.matches(binding.guard.correlation)
            or not node._busy or node._cancel.is_set()
            or node._goal_handles or getattr(node, '_pending_retained_acceptances', ())
            or node._held_book_corners is not None or node._target_book_model is not None
            or node._gripper_open_confirmed is not True):
        raise RuntimeError('release_pose_hold_context_changed')
    epoch = owner.evidence.release.open_epoch
    if (epoch is None or owner.open_wall is None or owner.last_ros is None
            or now < owner.last_ros or wall < owner.open_wall
            or now-epoch > 10_000_000_000 or wall-owner.open_wall > 30.):
        raise RuntimeError('release_pose_hold_clock_or_deadline')
    owner.last_ros = now
    if owner.evidence.fault:
        raise RuntimeError(owner.evidence.fault)
    latest_joint = owner.evidence.last_joint_stamp or epoch
    latest_contact = owner.evidence.last_raw_contact_stamp or epoch
    latest_odom = (getattr(node, '_delivery_raw_odom', None) or {}).get('stamp_ns')
    # Event topics do not promise initial bin messages. Positive-bin window
    # freshness is required by evaluate at ready/terminal, never from silence.
    if (now-latest_joint > 150_000_000 or not stamp_valid(latest_odom)
            or not -100_000_000 <= now-latest_odom <= 150_000_000):
        # Preserve the original diagnostic exception and locked scalar capture.
        raise ReleasePoseFeedbackStale(dict(
            now_ns=now, open_epoch_ns=epoch,
            latest_joint_stamp_ns=owner.evidence.last_joint_stamp,
            latest_raw_contact_stamp_ns=owner.evidence.last_raw_contact_stamp,
            latest_odom_stamp_ns=latest_odom,
            joint_age_ns=now-latest_joint, raw_contact_age_ns=now-latest_contact,
            odom_age_ns=now-latest_odom if stamp_valid(latest_odom) else None,
            joint_stale=now-latest_joint > 150_000_000,
            raw_contact_stale=now-latest_contact > 150_000_000,
            odom_stale=not stamp_valid(latest_odom) or not -100_000_000 <= now-latest_odom <= 150_000_000,
            stale_admission_inputs='continuous_joint_and_odom_only',
            freshness_limit_ns=150_000_000,
            raw_sample_sequence=owner.last_joint_sequence,
            raw_contact_frame_count=len(owner.evidence.raw_contact_frames),
            stationary_start_ns=owner.evidence.stationary_start,
            pending_acceptance_count=len(getattr(node, '_pending_retained_acceptances', ())),
            active_goal_count=len(node._goal_handles),
            evidence_fault=owner.evidence.fault,
            fault_latched=getattr(node, '_release_pose_fault_latched', None),
            owner_phase=owner.phase, cancelled=node._cancel.is_set()))
    raw_odom = node._delivery_raw_odom
    for name, limit in (('linear_speed', .005), ('angular_speed', .008)):
        value = raw_odom.get(name)
        if (not isinstance(value, (int, float)) or not math.isfinite(value)
                or not 0 <= value <= limit):
            raise RuntimeError('release_pose_latest_odom_not_stationary')


def _admit(node, owner, *, terminal=False, correlation=None):
    """Original scene predicate on a locked snapshot; reject a changed snapshot.

    The node sensor lock is not reentrant. A read-only proxy avoids taking it
    recursively; the exact source snapshot must still match under final lock.
    """
    with node._adaptive_command_guard():
        with node._lock:
            now, wall = int(node.get_clock().now().nanoseconds), time.monotonic()
            _require_owner_locked(node, owner, now, wall)
            captured = _scene_snapshot_locked(node)
        shadow = SimpleNamespace(_lock=nullcontext(), joints=captured[0],
            _joint_stamps_ns=captured[1], _staging_odom=captured[2],
            right_chain=node.right_chain, head_chain=node.head_chain,
            get_clock=node.get_clock)
        measured_scene_context(shadow, owner.binding.scene)
        with node._lock:
            now, wall = int(node.get_clock().now().nanoseconds), time.monotonic()
            _require_owner_locked(node, owner, now, wall)
            if _scene_snapshot_locked(node) != captured:
                return False
            ready = owner.evidence.evaluate(now, wall)['ready']
            if not ready:
                return False
            if terminal:
                if not owner.binding.identity.matches(correlation or {}):
                    raise RuntimeError('release_pose_terminal_attempt_changed')
                fields = owner.evidence.measurement(now, wall)
                fields.pop('event')
                # Linearize successful completion with raw-contact callbacks
                # and cancellation while both ownership locks remain held.
                owner.phase = 'publishing'
                node._publish_status(EVENT, **fields)
                node._publish_status('succeeded', command='place',
                    completion_mode=MODE, hand_return=False,
                    release_terminal_stamp_ns=now, **contract_fields(), **correlation)
                owner.phase = 'terminal'
            else:
                owner.phase = 'ready'
            return True


def finish(node, owner):
    if owner is None:
        raise RuntimeError('release_pose_finish_without_owner')
    try:
        while True:
            if _admit(node, owner):
                return True
            time.sleep(.002)
    except Exception:
        node._cancel.set()
        raise


def publish_terminal(node, correlation):
    owner = getattr(node, '_release_pose_owner', None)
    if owner is None:
        return False
    if not isinstance(owner, _Owner) or owner.phase != 'ready':
        raise RuntimeError('release_pose_terminal_without_completed_hold')
    if not _admit(node, owner, terminal=True, correlation=correlation):
        raise RuntimeError('release_pose_terminal_evidence_changed')
    return True


def cancel_postopen(node):
    with node._adaptive_command_guard():
        with node._lock:
            owner = getattr(node, '_release_pose_owner', None)
            if not isinstance(owner, _Owner) or owner.evidence.release.open_epoch is None:
                return False
            node._cancel.set()
            if owner.phase != 'terminal':
                owner.evidence.invalidate('release_pose_cancelled')
                node._publish_status('cancelled', command='place',
                    completion_mode=MODE, **vars(owner.binding.identity))
            return True


def deactivate(node):
    with node._adaptive_command_guard():
        with node._lock:
            owner = getattr(node, '_release_pose_owner', None)
            if (isinstance(owner, _Owner) and owner.evidence.release.open_epoch is not None
                    and owner.phase != 'terminal'):
                node._release_pose_fault_latched = owner.evidence.fault or 'release_pose_hold_failed'
                node._cancel.set()
            node._release_pose_owner = None
