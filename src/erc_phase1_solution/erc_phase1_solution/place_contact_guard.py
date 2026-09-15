"""Attempt-local arm/tool contact stop during scene-required PLACE.

Consumes observed scoped collision names, not poses or force estimates. The
first qualifying contact survives release and vanishing contact frames until
the command's finally block. It is not an official scored collision count.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import re


SCENE_MODELS = frozenset(('erc_collection_bin', 'erc_table'))
ARM_LINK = re.compile(r'arm_(left|right)_[1-7]_link\Z')
TOOL_LINK = re.compile(
    r'gripper_(left|right)_(base|base_finger_(left|right)|'
    r'inner_finger_(left|right)|outer_finger_(left|right)|'
    r'fingertip_(left|right)|screw_(left|right))_link\Z')


def _sensor_lock(node):
    return getattr(node, '_lock', None) or nullcontext()


def _robot_link(name):
    parts = str(name).split('::')
    for index, part in enumerate(parts[:-2]):
        if part == 'tiago_pro':
            link = parts[index + 1]
            if ARM_LINK.fullmatch(link) or TOOL_LINK.fullmatch(link):
                return link
    return None


def _scene_model(name):
    parts = str(name).split('::')
    return next((part for part in parts[:-2] if part in SCENE_MODELS), None)


def collision_pair(first, second):
    """Exact official arm/tool versus table/bin; intended book pairs excluded."""
    for robot, scene in ((first, second), (second, first)):
        link, model = _robot_link(robot), _scene_model(scene)
        if link is not None and model is not None:
            return dict(robot_link=link, scene_model=model,
                        collision_pair=[str(first), str(second)])
    return None


def producer_stamp(message):
    stamp = getattr(getattr(message, 'header', None), 'stamp', None)
    try:
        sec, nano = stamp.sec, stamp.nanosec
        if (isinstance(sec, bool) or isinstance(nano, bool)
                or int(sec) != sec or int(nano) != nano
                or sec < 0 or not 0 <= nano < 1_000_000_000):
            return None
        value = int(sec) * 1_000_000_000 + int(nano)
        return value if value > 0 else None
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


@dataclass
class PlaceContactGuard:
    started_ns: int
    correlation: dict
    fault: dict | None = None


def activate(node, correlation=None):
    if not getattr(node, 'table_scene_required', False):
        return
    with node._adaptive_command_guard():
        with _sensor_lock(node):
            node._place_contact_guard = PlaceContactGuard(
                int(node.get_clock().now().nanoseconds), dict(correlation or {}))


def deactivate(node):
    if getattr(node, '_place_contact_guard', None) is None:
        return
    # Wait for an already committed contact stop before allowing another command.
    with node._adaptive_command_guard():
        with _sensor_lock(node):
            node._place_contact_guard = None


def fault_reason(node):
    guard = getattr(node, '_place_contact_guard', None)
    return 'place_scene_contact' if guard is not None and guard.fault is not None else None


def require_clear(node):
    reason = fault_reason(node)
    if reason is not None:
        raise RuntimeError(reason)


def observe(node, message, source_topic='/contacts'):
    release_owner = getattr(node, '_release_pose_owner', None)
    if release_owner is not None:
        from .release_pose_finish import observe_contacts, post_open
        observe_contacts(node, message, source_topic)
    with _sensor_lock(node):
        guard = getattr(node, '_place_contact_guard', None)
    if guard is None or guard.fault is not None:
        return
    stamp = producer_stamp(message)
    # A queued pre-activation frame cannot create a new PLACE collision. An
    # exact hazardous pair with missing/malformed stamp stops conservatively,
    # explicitly without claiming a known producer time or freshness.
    if stamp is not None and stamp < guard.started_ns:
        return
    found = None
    for contact in getattr(message, 'contacts', ()):
        found = collision_pair(getattr(getattr(contact, 'collision1', None), 'name', ''),
                               getattr(getattr(contact, 'collision2', None), 'name', ''))
        if found is not None:
            break
    if found is None:
        return
    with node._adaptive_command_guard():
        with _sensor_lock(node):
            if getattr(node, '_place_contact_guard', None) is not guard or guard.fault is not None:
                return
            diagnostic = dict(found, source_topic=str(source_topic), producer_stamp_ns=stamp,
                              producer_stamp_valid=stamp is not None,
                              received_now_ns=int(node.get_clock().now().nanoseconds),
                              activation_ns=guard.started_ns)
            guard.fault = diagnostic
            node._cancel.set()
        # New contact event gets one current measured hold attempt, even if an
        # older close episode had already used its one-shot hold. No invented q.
        if release_owner is None or not post_open(node):
            node._adaptive_hold_sent = False
            try:
                node._hold_adaptive_gripper('place_scene_contact')
            except Exception as exc:
                diagnostic = dict(diagnostic, hold_error=str(exc))
        node._publish_status('payload_hazard', command='place',
                             reason='place_scene_contact', **diagnostic,
                             **guard.correlation)
