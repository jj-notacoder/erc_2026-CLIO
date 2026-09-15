"""Bounded current-command PLACE input/producer-idle handshake.

Fresh point and registered-scene admission stays in ManipulationNode. This
protocol only keeps the producer running until that admission succeeds, then
waits for the same request's applied idle acknowledgement before planning.
"""
from dataclasses import dataclass
import threading
import time
import uuid

REQUEST_FIELD = 'place_input_request_id'
IDLE_ACK_SECONDS = 2.0


@dataclass(frozen=True)
class InputRequest:
    trial_id: str
    request_id: str

    def fields(self):
        return dict(trial_id=self.trial_id, **{REQUEST_FIELD: self.request_id})


def request_from(payload):
    if not isinstance(payload, dict):
        return None
    trial, request = payload.get('trial_id'), payload.get(REQUEST_FIELD)
    if (type(trial) is not str or not 1 <= len(trial) <= 128
            or type(request) is not str or len(request) != 32
            or any(c not in '0123456789abcdef' for c in request)):
        return None
    return InputRequest(trial, request)


def command_fields(payload):
    """Preserve request correlation on success and all early refusal statuses."""
    if isinstance(payload, dict) and REQUEST_FIELD in payload:
        return dict(trial_id=payload.get('trial_id'),
                    **{REQUEST_FIELD: payload[REQUEST_FIELD]})
    return {}


def begin_request(mission, fields):
    request = InputRequest(mission.trial_id, uuid.uuid4().hex)
    mission._place_input_request = request
    fields[REQUEST_FIELD] = request.request_id


def finish_request(mission):
    """Idempotent producer cleanup on normal terminal, failure or cancellation."""
    if getattr(mission, '_place_input_request', None) is not None:
        mission._place_input_request = None
        mission._perception_mode('idle')


def mission_status(mission, payload):
    event = payload.get('event')
    if event != 'place_input_ready' and not (
            payload.get('command') == 'place'
            and event in ('succeeded', 'failed', 'rejected', 'cancelled')):
        return False
    request = request_from(payload)
    current = getattr(mission, '_place_input_request', None)
    if request is None or current != request:
        if event == 'place_input_ready' or current is not None:
            mission._log('place_input_ack_rejected', reason='request_mismatch', payload=payload)
            return True
        return False
    if event != 'place_input_ready':
        finish_request(mission)
        return False  # Preserve the original terminal/evidence handling.
    if mission.finished or mission.state != 'PLACE' or payload.get('command') != 'place':
        mission._log('place_input_ack_rejected', reason='inactive_place', payload=payload)
        return True
    if getattr(mission, 'delivery_evidence_enabled', False):
        evidence = getattr(mission, 'delivery_release_evidence', None)
        if evidence is None or not evidence.identity.matches(payload):
            mission._log('place_input_ack_rejected', reason='attempt_mismatch', payload=payload)
            return True
    # _command supplies the current trial id. The perception acknowledgement
    # is emitted after idle is applied, and the manipulation waiter owns it.
    mission._command(mission.mode_pub, 'idle',
        shelf_column_number=mission.target_column, book_colour=mission.target_colour,
        **{REQUEST_FIELD: request.request_id})
    mission._log('place_input_captured', payload=payload)
    return True


def perception_idle(perception, payload):
    """Called after the original mode handler has applied idle and reset state."""
    request = request_from(payload)
    if request is not None and payload.get('event') == 'idle' and perception.mode == 'idle':
        perception._publish_status('place_input_idle', **request.fields())


def observe_idle(node, payload):
    if payload.get('event') != 'place_input_idle' or payload.get('mode') != 'idle':
        return
    request = request_from(payload)
    with node._lock:
        waiter = getattr(node, '_place_input_waiter', None)
        if (request is not None and waiter is not None and waiter[0] == request
                and not node._cancel.is_set()):
            waiter[1].set()


def await_idle_after_capture(node, payload):
    """Called only after the existing fresh input and workspace checks pass."""
    if not isinstance(payload, dict) or REQUEST_FIELD not in payload:
        return  # Direct legacy commands retain their existing behavior.
    request = request_from(payload)
    if request is None:
        raise RuntimeError('place_input_request_invalid')
    waiter = (request, threading.Event())
    with node._lock:
        if node._cancel.is_set():
            raise RuntimeError('place_input_handoff_cancelled')
        if getattr(node, '_place_input_waiter', None) is not None:
            raise RuntimeError('place_input_handoff_already_active')
        node._place_input_waiter = waiter
    try:
        fields = request.fields()
        fields.update({key: payload[key] for key in ('placement_attempt_id', 'target_model')
                       if key in payload})
        node._publish_status('place_input_ready', command='place', **fields)
        deadline = time.monotonic() + IDLE_ACK_SECONDS
        while not node._cancel.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError('place_input_idle_ack_timeout')
            if waiter[1].wait(min(.02, remaining)):
                if node._cancel.is_set():
                    raise RuntimeError('place_input_handoff_cancelled')
                return
        raise RuntimeError('place_input_handoff_cancelled')
    finally:
        with node._lock:
            if getattr(node, '_place_input_waiter', None) is waiter:
                node._place_input_waiter = None
