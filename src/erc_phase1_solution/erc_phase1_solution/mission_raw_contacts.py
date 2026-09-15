"""Synchronous complete Contacts decoding for the mission's existing callbacks.

The ordinary ROS callback group serializes these subscriptions and mission
ticks. No queue, thread, contact filtering or synthetic heartbeat is added.
"""
from dataclasses import asdict, dataclass
import hashlib
import time

from .raw_contacts import decode_contacts_cdr


@dataclass(frozen=True)
class DecodeFailure:
    reason: str
    source_topic: str
    error_type: str
    error: str
    serialized_bytes: int
    raw_sha256: str
    raw_prefix_hex: str
    raw_prefix_truncated: bool


def checked_enabled(value):
    if type(value) is not bool:
        raise ValueError('mission_raw_contacts_enabled must be Boolean')
    return value


def _fail_closed(node, data, error, source_topic):
    """Retain the first wire fault and use the ordinary mission abort path.

    A malformed message has no usable producer epoch or contact verdict.
    Do not deliver a fabricated empty message or convert it with a fallback.
    A diagnostic received after a finished trial cannot rewrite its outcome.
    """
    if node._mission_raw_contacts_failure is not None:
        return
    raw = data if type(data) is bytes else b''
    reason = 'raw_contacts_decode_failed'
    failure = DecodeFailure(reason, source_topic, type(error).__name__,
        str(error)[:256], len(raw), hashlib.sha256(raw).hexdigest(),
        raw[:256].hex(), len(raw) > 256)
    node._mission_raw_contacts_failure = failure
    try:
        if not node.finished:
            try:
                evidence = getattr(node, 'delivery_release_evidence', None)
                if evidence is not None:
                    evidence.invalidate(reason)
                    node._delivery_release_snapshot = evidence.evaluate(
                        int(node.get_clock().now().nanoseconds), time.monotonic())
            finally:
                try:
                    node._abort(reason)
                except Exception:
                    # _abort commits finished before publishing. If a failed
                    # publication interrupted it, attempt both existing cancel
                    # destinations independently, persist failure, then retain
                    # the original exception. Never resume ordinary callbacks.
                    node.finished = True
                    node.state = 'ABORTED'
                    node._place_input_request = None
                    for publisher in (node.nav_command_pub, node.manip_command_pub):
                        try:
                            node._command(publisher, 'cancel')
                        except Exception:
                            pass
                    try:
                        node._write_summary(False, reason)
                    except Exception:
                        pass
                    raise
    finally:
        try:
            node._log('sensor_fault', **asdict(failure))
        except Exception:
            pass  # Failure latch/abort cannot depend on diagnostic publication.


def mission_raw_contacts_callback(node, callback, source_topic):
    if source_topic not in ('/contacts', '/bin_contacts'):
        raise ValueError('Unsupported mission Contacts topic')

    def receive(data):
        if node._mission_raw_contacts_failure is not None:
            return None
        try:
            message = decode_contacts_cdr(data)
        except Exception as error:
            _fail_closed(node, data, error, source_topic)
            return None
        # Preserve exceptions and all original state/clock/target processing.
        # Callback exceptions are not mislabeled as decoder failures.
        return callback(message)
    return receive
